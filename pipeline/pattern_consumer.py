import configparser
import fcntl
import itertools
import json
import os
from pathlib import Path
import pickle
import socket
import time

import bidict
import pandas as pd

import threading
from queue import Queue, Empty
import re

CURRENT_TREE_ID_SUFFIX = "_current_tree_id.txt"
SUBTREE_PARALLELISM = 128   


def save_partition_tree_pkl(output_prefix: str, suffix_id: int, trees: list) -> str:
    pkl_path = f"{output_prefix}_partition_{suffix_id}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({"partition_id": suffix_id, "trees": trees}, f)
        f.flush()
        os.fsync(f.fileno())
    return pkl_path


def write_current_tree_id(output_prefix: str, suffix_id: int):
    final_path = output_prefix + CURRENT_TREE_ID_SUFFIX
    tmp_path = final_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(str(suffix_id))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)   

PARTITION_QUEUE_SUFFIX = "_pending_partitions.txt"
PRODUCER_INITIAL_WAIT = 1
PRODUCER_POLL_INTERVAL = 0.5

def queue_file_has_content(queue_filepath: str) -> bool:
    if not os.path.exists(queue_filepath):
        return False
    try:
        with open(queue_filepath, 'r', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    if line.strip():
                        return True
                return False
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        return False


def pop_top_partition_name(queue_filepath: str):
    if not os.path.exists(queue_filepath):
        return None
    try:
        with open(queue_filepath, 'a+', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                lines = [l.strip() for l in f if l.strip()]
                if not lines:
                    return None
                top = lines[0]
                f.seek(0)
                f.truncate()
                for l in lines[1:]:
                    f.write(l + '\n')
                f.flush()
                os.fsync(f.fileno())
                return top
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        return None


def read_partition_row_ids(partition_filepath: str) -> list[int]:
    ids = []
    with open(partition_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(line))
    return ids


def parse_partition_suffix_id(partition_name: str) -> int:
    m = re.search(r"_partition_(\d+)\.txt$", partition_name)
    return int(m.group(1)) if m else -1

class IncrementalPatternReader:
    def __init__(self, output_prefix: str):
        self.output_prefix = output_prefix
        self.pattern_to_id_bidict = bidict.bidict()
        self.pattern_to_benefit_dict = dict()
        self.file_offsets = {}

    def read_new_patterns(self) -> int:
        base_dir_path = str(Path(self.output_prefix).parent)
        new_count = 0
        for f in Path(base_dir_path).rglob("*"):
            filepath = str(f)
            if not (f.is_file() and filepath.startswith(self.output_prefix + '_mining_F')):
                continue
            offset = self.file_offsets.get(filepath, 0)
            try:
                with open(filepath, 'r') as fh:
                    fcntl.flock(fh, fcntl.LOCK_SH)
                    try:
                        fh.seek(offset)
                        while True:
                            line = fh.readline()
                            if not line:
                                break                 
                            if not line.endswith('\n'):
                                break                 
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                break
                            pattern_frozenset = frozenset(data["pattern"])
                            if pattern_frozenset not in self.pattern_to_id_bidict:
                                self.pattern_to_id_bidict[pattern_frozenset] = data["id"]
                                self.pattern_to_benefit_dict[data["id"]] = data["benefit"]
                                new_count += 1
                            self.file_offsets[filepath] = fh.tell()
                    finally:
                        fcntl.flock(fh, fcntl.LOCK_UN)
            except FileNotFoundError:
                continue
        return new_count


class TreeNode:
    def __init__(self, values: list[str] = None, childs: list = None, row_id: int = None):
        self.values = values  
        self.childs = childs  
        self.row_id = row_id

    def __repr__(self):
        return "Node(" + str(self.values) + ")"
    
    def remove_cols(self, shared_cols: frozenset[str]):
        self.values = list(filter(lambda x: x not in shared_cols, self.values))
        
    def relax_node(self): 
        if len(self.values) > 1:
            child_childs = self.childs
            child_values = self.values[1:] 
            child_row_id = self.row_id
            
            self.values = [self.values[0]]
            self.childs = [TreeNode(values=child_values, childs=child_childs, row_id=child_row_id)]
            self.row_id = None
    
        if self.childs is not None:
            for child in self.childs:
                child.relax_node()
                
    def clean_node(self):
        return
            
    def cleanup(self):
        if self.childs is not None:
            for child in self.childs:
                if child is not None:
                    child.cleanup()
        
        self.values = None
        self.childs = None
        self.row_id = None

class GlobalTreeBuilder:
    def __init__(
        self,
        pattern_to_id_bidict: bidict.bidict[frozenset[str], int],
        pattern_to_benefit_dict: dict[int, int],
        dataset: pd.DataFrame,
        output_prefix: str,
        row_ids: list[int] = None   
    ):
        self.pattern_to_id_bidict = pattern_to_id_bidict
        self.pattern_to_benefit_dict = pattern_to_benefit_dict
        self.output_prefix = output_prefix
        self.trees = []
        rows_excluded_set = set()

        if row_ids is None:
            row_iter = enumerate(dataset.to_dict('records'))
        else:
            sub_df = dataset.iloc[row_ids]
            row_iter = zip(row_ids, sub_df.to_dict('records'))

        for row_id, row_dict in row_iter:
            if row_id in rows_excluded_set:
                continue
            formatted_row = []
            for column_name, value in row_dict.items():
                formatted_row.append(f"{column_name}: _{value}")
            self.trees.append(TreeNode(values=formatted_row, row_id=row_id, childs=None))
        
   
    def __repr__(self):
        return "Global Tree Group"
    
    def relax_trees(self):
        for i, tree in enumerate(self.trees):
            tree.relax_node()
            if i % 1000 == 0 and i > 0:
                pass
        
    def clean_nodes(self):
        for tree in self.trees:
            tree.clean_node()
            
    def merge_trees(self, shared_prefix_dict: dict[int, list[int]]):
        merged_tree_indexes = []
        for k, v in shared_prefix_dict.items():
            if len(v) == 0: 
                continue
            pattern = self.pattern_to_id_bidict.inv[k]
            values = list(pattern) 
            childs = [self.trees[i] for i in v] 
            row_id = None
            merged_tree_indexes.extend(v)
            
            for child in childs:
                child.remove_cols(pattern)
            
            self.trees.append(TreeNode(values=values, childs=childs, row_id=row_id))
            
        merged_tree_indexes_set = set(merged_tree_indexes)
        self.trees = [x for i, x in enumerate(self.trees) if i not in merged_tree_indexes_set]
    
    def next_iteration(self):
        pattern_to_tree_dict = {}
        tree_to_pattern_dict = {}
        can_merge = False 
        
        for tree_id, root in enumerate(self.trees):
            fullset = root.values
            assert type(fullset) is list
            
            all_subsets = []
            for r in range(1, len(fullset) + 1):
                all_subsets.extend(list(itertools.combinations(fullset, r)))
            all_subsets = [frozenset(subset) for subset in all_subsets]

            for subset in all_subsets:
                if subset in self.pattern_to_id_bidict: 
                    pattern_id = self.pattern_to_id_bidict[subset]
                    if pattern_id in pattern_to_tree_dict:
                        pattern_to_tree_dict[pattern_id].append(tree_id)
                        can_merge = True 
                    else:
                        pattern_to_tree_dict[pattern_id] = [tree_id]
                
                    if tree_id in tree_to_pattern_dict:
                        tree_to_pattern_dict[tree_id].append(pattern_id)
                    else:
                        tree_to_pattern_dict[tree_id] = [pattern_id]
        
        if can_merge is False:
            return can_merge
        
        pattern_to_tree_dict = {k: v if len(v) > 1 else [] for k, v in pattern_to_tree_dict.items()}
                     
        pattern_to_tree_dict = dict(
            sorted(
                pattern_to_tree_dict.items(),
                key=lambda item: self.pattern_to_benefit_dict[item[0]]
            )
        )
        
        shared_prefix_dict = {}
        merged_signs = [False] * len(self.trees)
        for k, v in reversed(pattern_to_tree_dict.items()): 
            shared_prefix_dict[k] = [] 
            for tree_id in v:
                if merged_signs[tree_id] is False:
                    shared_prefix_dict[k].append(tree_id)
                    merged_signs[tree_id] = True            
                      
        shared_prefix_dict = dict(
            sorted(
                shared_prefix_dict.items(),
                key=lambda item: self.pattern_to_benefit_dict[item[0]]
            )
        )

        for k, v in shared_prefix_dict.items():  
            if len(v) == 0:
                continue
            t_a = self.pattern_to_benefit_dict[k]
            n_a = len(v)
            avg_benefit = 1.0 * t_a * (n_a - 1) / n_a
            moved_dict = {}
            all_rows_have_target = True
            for tree_id in v:
                target_pattern_id = None
                max_benefit = -10000000
                for pattern_id in tree_to_pattern_dict[tree_id]:
                    if pattern_id == k:
                        continue
                    t_b = self.pattern_to_benefit_dict[pattern_id]
                    if t_b < t_a and \
                       t_b > avg_benefit and \
                       len(shared_prefix_dict[pattern_id]) > 0:
                        if t_b > max_benefit:
                            max_benefit = t_b
                            target_pattern_id = pattern_id
                if target_pattern_id is None:
                    all_rows_have_target = False
                    break  
                if target_pattern_id in moved_dict:
                    moved_dict[target_pattern_id].append(tree_id)
                else:
                    moved_dict[target_pattern_id] = [tree_id]
            if not all_rows_have_target:
                continue  
            shared_prefix_dict[k] = []
            for moved_k, moved_v in moved_dict.items():
                shared_prefix_dict[moved_k].extend(moved_v)

        self.merge_trees(shared_prefix_dict)
        return can_merge 

    
    def build_trees(self):
        iteration_count = 0 
        
        can_merge = self.next_iteration()
        iteration_count += 1
        
        while can_merge:
            can_merge = self.next_iteration()
            iteration_count += 1
    
        self.relax_trees()

class TreeIterator:
    def __init__(self, root: TreeNode, cols_num: int):
        self.child_indexes = [] 
        self.visiting_nodes = [] 
        self.root = root
        self.cols_num = cols_num
        self.iter_count = 0
        
        node = self.root
        while True:
            if node is not None:
                self.child_indexes.append(0)
                self.visiting_nodes.append(node)
               
                if node.childs is None or len(node.childs) == 0:
                    node = None
                else:
                    node = node.childs[0]
            else:
                break
                
    def next_unprocessed_row(self) -> tuple[list[str], list[str]]:
        return self.next_row()
    
    def next_row(self) -> tuple[list[str], list[str]]:
        output_row = []
        output_col = []
        
        if len(self.visiting_nodes) > 0:
            current_row_id = self.visiting_nodes[-1].row_id
            
            output_values = [] 
            for _, node in enumerate(self.visiting_nodes):
                output_values.extend(node.values)
                
            output_values.append(f"row_id: _{current_row_id}") 
                
            assert ": _" in output_values[0] 
            output_row = [value.split(": _", 1)[1] for value in output_values] 
            output_col = [value.split(": _", 1)[0] for value in output_values]
            
            assert len(output_row) == self.cols_num
            
            self.iter_count += 1
            if self.iter_count % 500 == 0:
                pass
            else:
                pass
        else:
            return output_row, output_col 
        
        assert len(self.visiting_nodes) == len(self.child_indexes)
        
        while len(self.visiting_nodes) > 0:
            last_idx = len(self.visiting_nodes) - 1
            
            if last_idx >= len(self.child_indexes) - 1:
                popped_node = self.visiting_nodes.pop()
            elif self.child_indexes[last_idx+1] >= len(self.visiting_nodes[last_idx].childs) - 1:
                popped_node = self.visiting_nodes.pop()
                self.child_indexes.pop()
            else:
                break  
                
        if len(self.visiting_nodes) == 0:
            self.child_indexes.clear()
            return output_row, output_col 
                
        next_child_idx = self.child_indexes[-1] + 1
        assert len(self.child_indexes) == len(self.visiting_nodes) + 1
        self.child_indexes.pop()
        assert len(self.child_indexes) == len(self.visiting_nodes)
        
        node = self.visiting_nodes[-1].childs[next_child_idx]
        
        while True:
            if node is not None:
                self.child_indexes.append(next_child_idx)
                self.visiting_nodes.append(node)
               
                if node.childs is None or len(node.childs) == 0:
                    node = None
                else:
                    next_child_idx = 0
                    node = node.childs[0]
            else:
                break
        
        return output_row, output_col
    
    def is_end(self):
        result = len(self.visiting_nodes) == 0
        if result:
            pass
        return result

class TreeIteratorGroup:
    def __init__(self, roots: list[TreeNode], cols_num: int, trees_limit: int = 1):
        self.roots = roots
        self.cols_num = cols_num
        self.trees_limit = max(1, int(trees_limit))   
        self.iterators = []
        self.next_tree_idx = 0
        
        while not self.is_end() and not self.is_full():
            self.insert_iterator()
            
    def reset(self):
        self.iterators = []
        self.next_tree_idx = 0
        
        while not self.is_end() and not self.is_full():
            self.insert_iterator()
            
    def is_end(self):
        result = len(self.roots) <= self.next_tree_idx
        if result and len(self.roots) > 0: 
            pass
        return result
    
    def is_full(self):
        result = len(self.iterators) >= self.trees_limit
        if result:
            pass
        return result
        
    def insert_iterator(self):
        new_iterator = None
        while True:
            if self.is_end() or self.is_full():
                return
            
            new_iterator = TreeIterator(
                self.roots[self.next_tree_idx],
                self.cols_num
            )
            if not new_iterator.is_end():
                self.iterators.append(new_iterator)
            else:
                pass
                
            self.next_tree_idx = self.next_tree_idx + 1
        
    def clean_iterator(self):
        before_count = len(self.iterators)
        self.iterators = [iter for iter in self.iterators if not iter.is_end()]
        after_count = len(self.iterators)
        
        if before_count != after_count:
            pass
        
    def return_row_batch(self, max_rows: int = None):
        output_rows = []
        output_cols = []
        for iter in self.iterators:
            if max_rows is not None and len(output_rows) >= max_rows:
                break
            row, col = iter.next_unprocessed_row()
            assert len(row) == len(col)
            if len(row) > 0:
                output_rows.append(row)
            if len(col) > 0:
                output_cols.append(col)
        self.clean_iterator()
        while not self.is_end() and not self.is_full():
            self.insert_iterator()
        return output_rows, output_cols

    def already_finished(self):
        result = self.is_end() and len(self.iterators) == 0
        if result:
            pass
        return result

def partition_tree_producer(dataset: pd.DataFrame,
                            output_prefix: str,
                            tree_cache: Queue,
                            miner_finished: threading.Event):
    queue_filepath = output_prefix + PARTITION_QUEUE_SUFFIX
    mining_end_path = output_prefix + "_mining.end"
    base_dir = os.path.dirname(os.path.abspath(output_prefix))

    time.sleep(PRODUCER_INITIAL_WAIT)

    try:
        write_current_tree_id(output_prefix, 0)
    except Exception as e:
        pass

    pattern_reader = IncrementalPatternReader(output_prefix)

    while True:
        mining_ended = os.path.exists(mining_end_path)

        if not tree_cache.empty():
            if mining_ended and not queue_file_has_content(queue_filepath):
                miner_finished.set()
                return
            time.sleep(PRODUCER_POLL_INTERVAL)
            continue

        part_name = pop_top_partition_name(queue_filepath)
        if part_name is None:
            if mining_ended:
                miner_finished.set()
                return
            time.sleep(PRODUCER_POLL_INTERVAL)
            continue

        part_path = os.path.join(base_dir, part_name)
        if not os.path.exists(part_path):
            continue

        row_ids = read_partition_row_ids(part_path)
        suffix_id = parse_partition_suffix_id(part_name)

        if not row_ids:
            continue

        n_new = pattern_reader.read_new_patterns()

        builder = GlobalTreeBuilder(
            pattern_to_id_bidict=pattern_reader.pattern_to_id_bidict,
            pattern_to_benefit_dict=pattern_reader.pattern_to_benefit_dict,
            dataset=dataset,
            output_prefix=output_prefix,
            row_ids=row_ids
        )
        
        builder.build_trees()
        try:
            pkl_path = save_partition_tree_pkl(output_prefix, suffix_id, builder.trees)
        except Exception as e:
            pass
        tree_cache.put((suffix_id, builder.trees))




def prompt_serving(dataset_path: str, output_prefix: str, dataset_type: str,
                   columns_list: list[str], is_building_tree: bool,
                   tree_building_interval: int, cols_num: int,
                   host: str, port: int, prompt_prefix: str):
    dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
    dataset = dataset[columns_list]

    tree_cache = Queue(maxsize=1)        
    miner_finished = threading.Event()   
    producer = threading.Thread(
        target=partition_tree_producer,
        args=(dataset, output_prefix, tree_cache, miner_finished),
        daemon=True, name="partition-tree-producer")
    producer.start()

    prompt_file_id = 0
    serve_start_time = time.time()
    total_sent_rows = 0
    TIMER_INTERVAL = 5000

    serve_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serve_socket.bind((host, port))
    serve_socket.listen(1)

    current_group = None           
    current_partition_id = None

    def _all_done() -> bool:
        return (miner_finished.is_set()
                and tree_cache.empty()
                and (current_group is None or current_group.already_finished()))

    should_finished = False
    try:
        while True:
            if should_finished:
                break
            client_socket, client_address = serve_socket.accept()
            try:
                data = client_socket.recv(4)
                if not data:
                    continue
                req_num = int.from_bytes(data, byteorder='big')

                res_str = str()
                if _all_done():
                    res_str = "already end"
                else:
                    prompt_file_id += 1   
                    prompt_filepath = prompt_prefix + "_prompt_" + str(prompt_file_id) + ".jsonl"
                    with open(prompt_filepath, 'w') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        written_rows = 0
                        try:
                            while req_num > 0:
                                if current_group is None or current_group.already_finished():
                                    got = False
                                    while not got:
                                        try:
                                            current_partition_id, trees = tree_cache.get(timeout=0.2)
                                            current_group = TreeIteratorGroup(trees, cols_num, trees_limit=SUBTREE_PARALLELISM)
                                            try:
                                                write_current_tree_id(output_prefix, current_partition_id)
                                            except Exception as e:
                                                pass
                                            got = True
                                        except Empty:
                                            if _all_done():
                                                break     
                                            if written_rows > 0:
                                                break     
                                            time.sleep(0.05)
                                    if not got:
                                        break

                                rows, cols = current_group.return_row_batch(max_rows=req_num)
                                assert len(rows) == len(cols)
                                if len(rows) == 0:                
                                    if _all_done():
                                        break
                                    time.sleep(0.05)
                                    continue
                                for i in range(len(rows)):        
                                    df = pd.DataFrame(data=[rows[i]], columns=cols[i])
                                    row_ids = df.pop('row_id').tolist()
                                    assert len(row_ids) == 1
                                    cols_list = df.columns.tolist()
                                    assert 'row_id' not in cols_list
                                    prompt_data_part = str()
                                    for col in cols_list:
                                        prompt_data_part += ("{" + col + "} -> " + json.dumps(
                                            df[col].astype(str).tolist(), ensure_ascii=False) + "\n\n")
                                    prompt_part = {"data_part": prompt_data_part,
                                                "row_id_pair_part": {"row_id": row_ids[0]}}
                                    f.write(json.dumps(prompt_part, ensure_ascii=False) + '\n')
                                    written_rows += 1
                                    req_num -= 1
                                    total_sent_rows += 1
                                if total_sent_rows % TIMER_INTERVAL == 0:
                                    elapsed = time.time() - serve_start_time
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                    if written_rows > 0:
                        res_str = prompt_filepath
                    elif _all_done():
                        res_str = "already end"

                if res_str == "already end":
                    should_finished = True
                client_socket.send(res_str.encode('utf-8'))
            finally:
                client_socket.close()
    except KeyboardInterrupt:
        serve_socket.close()
    finally:
        elapsed = time.time() - serve_start_time
        end_signal_path = output_prefix + "_tree.end"
        open(end_signal_path, 'w').close()
        miner_finished.set()
        producer.join(timeout=5)
        serve_socket.close()
    return



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config/tree_eval.ini', encoding='utf-8')
    
    columns_name = config.get('backend', 'columns_name')
    columns_list = [col.strip() for col in columns_name.split(',')]
    tree_building_interval = config.getint('backend', 'build_interval')
    is_building_tree = config.getboolean('backend', 'building_tree')
    dataset_path = config.get('backend', 'dataset_path')
    output_prefix = config.get('backend', 'output_prefix')
    dataset_type = config.get('backend', 'dataset_type')
    cols_num = config.getint('backend', 'cols_num')
    host = config.get('backend', 'host')
    port = config.getint('backend', 'port')
    prompt_prefix = config.get('backend', 'prompt_prefix')
    
    prompt_serving(dataset_path, output_prefix, dataset_type, columns_list, is_building_tree, tree_building_interval, cols_num+1, host, port, prompt_prefix)