from datetime import datetime
import heapq
from bitarray import bitarray
import numpy as np          
import pandas as pd
from transformers import AutoTokenizer
import json
from multiprocessing import Process
import configparser
import fcntl
import glob
import os
import itertools   

pattern_id = 0
CUSUM_SAMPLE_SIZE = 3000
CUSUM_K_SIGMA = 0.5
CUSUM_H_SIGMA = 5.0
CUSUM_MAX_WAIT_SECONDS = None   
PARTITION_QUEUE_NAME = "_pending_partitions.txt"
PARTITION_CD = 15
PARTITION_QUEUE_CAPACITY = 1
SUB_PATTERN_DEPTH_L = 3
ROW_SCALE_THRESHOLD_S = 3000
ENABLE_ROW_DEDUP = False

def convert_to_list(source: pd.DataFrame) -> list[list[str]]:
    formatted_rows = []
    for row_dict in source.to_dict('records'):
        formatted_row = []
        for column_name, value in row_dict.items():
            formatted_row.append(f"{column_name}: _{value}")
        formatted_rows.append(formatted_row)
    return formatted_rows

class CUSUMHighPotentialMonitor:
    def __init__(self,
                 sample_size: int = CUSUM_SAMPLE_SIZE,
                 k_sigma: float = CUSUM_K_SIGMA,
                 h_sigma: float = CUSUM_H_SIGMA,
                 max_wait_seconds: float = CUSUM_MAX_WAIT_SECONDS):
        self.sample_size = int(sample_size)
        self.k_sigma = float(k_sigma)
        self.h_sigma = float(h_sigma)
        self.max_wait_seconds = max_wait_seconds
        self.samples = []                    
        self.sample_kinds = []   
        self.mu0 = None                      
        self.sigma = None                    
        self.M = None                      
        self.k = None                        
        self.h = None                       
        self.S = 0.0                         
        self.t = 0                          
        self.end_t = None                    
        self.high_potential_phase_ended = False   
        self._monitor_start_ts = None       

    def __finish_sampling(self):
        arr = np.asarray(self.samples, dtype=np.float64)
        n = arr.size
        self.mu0 = float(arr.mean())                 
        self.M = float(np.median(arr))               
        mad = float(np.median(np.abs(arr - self.M)))  
        self.sigma = 1.4826 * mad                     
        if self.sigma <= 0.0:
            self.sigma = max(abs(self.mu0) * 0.01, 1)
        self.k = self.k_sigma * self.sigma           
        self.h = self.h_sigma * self.sigma            
        self.S = 0.0
        self.t = 0
        self._monitor_start_ts = datetime.now()
        self.samples = []                            

    def __mark_ended(self, reason: str):
        self.high_potential_phase_ended = True      
        self.end_t = self.t
        elapsed = 0.0
        if self._monitor_start_ts is not None:
            elapsed = (datetime.now() - self._monitor_start_ts).total_seconds()

    def observe(self, potential: float, sample_kind: str = "own") -> bool:
        if self.high_potential_phase_ended:
            return False

        if self.mu0 is None:
            self.samples.append(float(potential))
            self.sample_kinds.append(sample_kind)  
            if len(self.samples) >= self.sample_size:
                self.__finish_sampling()
            return False

        self.t += 1
        self.S = max(0.0, self.S + (self.mu0 - float(potential)) - self.k)

        if (self.max_wait_seconds is not None
                and self._monitor_start_ts is not None
                and (datetime.now() - self._monitor_start_ts).total_seconds() >= self.max_wait_seconds):
            self.__mark_ended(reason=f"等待超时 (>= {self.max_wait_seconds}s)")
            return True

        if self.S > self.h:
            self.__mark_ended(reason=f"S_t={self.S:.4f} > h={self.h:.4f}")
            return True

        return False

class PartitionQueueFile:
    def __init__(self, path: str):
        self.path = path
        self.recreate()

    def recreate(self):
        with open(self.path, 'a+', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                f.truncate()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def pending_names(self) -> list[str]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, 'r', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                return [line.strip() for line in f if line.strip()]
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def pending_count(self) -> int:
        return len(self.pending_names())

    def push(self, name: str) -> bool:
        with open(self.path, 'a+', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                names = [line.strip() for line in f if line.strip()]
                if len(names) >= PARTITION_QUEUE_CAPACITY:
                    return False
                f.write(name + '\n')
                f.flush()
                os.fsync(f.fileno())
                return True
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def push_unconditional(self, name: str):
        with open(self.path, 'a', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(name + '\n')
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

class PatternNode:
    def __init__(self, items, token_length, count):
        self.items = list(items)               
        self.item_key = frozenset(items)       
        self.token_length = int(token_length)  
        self.row_count = int(count)            
        self.unextracted_mask = None           
        self.unextracted_rows = None 
        self.unextracted_count = int(count)    
        self.sub_patterns = []                 
        self.score = 0.0                       

    def __repr__(self):
        return (f"PatternNode(items={self.items}, tokens={self.token_length}, "
                f"rows={self.row_count}, score={self.score:.4f})")

class PatternRegistry:
    def __init__(self, item_masks=None, num_rows=0):
        self.key_to_node = dict()      
        self.item_masks = item_masks   
        self.num_rows = num_rows       
        self.cusum_monitor = CUSUMHighPotentialMonitor()

        self.pattern_write_count = 0
        self.own_pattern_write_count = 0
        self.absorbed_pattern_write_count = 0
        self.on_high_potential_ended = None

    def register(self, items, token_length, count) -> PatternNode:
        key = frozenset(items)
        existed = self.key_to_node.get(key)
        if existed is not None:
            return existed
        node = PatternNode(items, token_length, count)

        if self.item_masks:
            mask = bitarray(self.num_rows)
            mask.setall(1)
            for it in node.items:
                mask &= self.item_masks[it]
            node.row_count = mask.count()
            node.unextracted_count = node.row_count

            if node.row_count * 32 < self.num_rows:
                buf  = np.frombuffer(mask.tobytes(), dtype=np.uint8)
                bits = np.unpackbits(buf, bitorder='big')[:self.num_rows]
                node.unextracted_rows = np.flatnonzero(bits).astype(np.int32)
                node.unextracted_mask = None          
            else:
                node.unextracted_mask = mask          
                node.unextracted_rows = None

        self.key_to_node[key] = node

        for it in node.items:
            sub_node = self.key_to_node.get(key - frozenset([it]))
            if sub_node is not None:
                node.sub_patterns.append(sub_node)
        return node

def write_file_with_handle(writter, patterns: list[list[str]], tokenizer, counts,
                           pattern_registry: PatternRegistry = None,
                           kind: str = "own"):
    global pattern_id
    assert len(patterns) == len(counts)
    for i in range(len(patterns)):
        pattern = patterns[i]
        count = counts[i]
        benefit = 0
        for col in pattern:
            benefit += len(tokenizer.encode(col.split(": _", 1)[1]))
        pattern_id += 1
        potential = benefit * (count - 1) / count
        item = {"pattern": pattern, "benefit": benefit, "id": pattern_id,
                "count": count, "potential": potential}
        json_str = json.dumps(item, ensure_ascii=False) + '\n'
        fcntl.flock(writter.fileno(), fcntl.LOCK_EX)
        try:
            writter.write(json_str)
            writter.flush()   
        finally:
            fcntl.flock(writter.fileno(), fcntl.LOCK_UN)

        if pattern_registry is not None:
            pattern_registry.register(pattern, benefit, count)
            pattern_registry.pattern_write_count += 1
            if kind == "absorbed":
                pattern_registry.absorbed_pattern_write_count += 1
            else:
                pattern_registry.own_pattern_write_count += 1
            monitor = pattern_registry.cusum_monitor
            if monitor is not None and not monitor.high_potential_phase_ended:
                just_ended = monitor.observe(item["potential"], sample_kind=kind)
                if just_ended and pattern_registry.on_high_potential_ended is not None:
                    pattern_registry.on_high_potential_ended()



def write_mining_end_signal(output_prefix: str):
    signal_path = output_prefix + "_mining.end"
    open(signal_path, 'w').close()


class BMCTreeNode:
    def __init__(self, item, count, bitmap_code):
        self.item = item
        self.count = count
        self.bitmap_code = bitmap_code
        self.children = dict()

    def get_child_registering_item(self, item):
        return self.children.get(item)

    def add_child(self, child):
        self.children[child.item] = child

    def __repr__(self):
        return f'{self.item}:{self.count}->{self.bitmap_code}'


class FrequentItemsetTreeNode:
    def __init__(self):
        self.item = None
        self.count = 0
        self.children = []
        self.NegNodeSet = []
        self.pattern_idx = None      
        self.parent = None            
        self.siblings = None       
        self.sib_start = 0          
        self.inherited_absorbed = ()  
        self.own_absorbed = None      
        self.token_length = 0        
        self.potential = 0.0         
        self.cur_count = None   

    def __repr__(self):
        return f'{self.item}'


def clean_BMC_tree(root):
    for item, child in root.children.items():
        clean_BMC_tree(child)
    del root.item
    del root.children


class NegFIN:
    def __init__(self, source_df, filepath_prefix, alpha, llm_path, task_id):
        self.dataset = convert_to_list(source_df)
        self.alpha = alpha
        self.min_count = 2
        self.filepath_prefix = filepath_prefix
        self.num_of_transactions = None
        self.F1 = None
        self.item_to_NodeSet = None
        self.writer_list = []
        self.filename_list = []
        self.num_of_frequent_itemsets = 0;
        self.execution_time = None
        self.task_id = task_id
        
        self.item_token_len_cache = {}  
        self.source_df = source_df   
        self.dup_row_map = None   
        if ENABLE_ROW_DEDUP:
            self.__deduplicate_rows()
        else:
            pass

        self.pattern_registry = None  
        self.item_row_masks = None        
        self.output_row_mask_np = None   
        self.extract_time_records = []    

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        self.partition_suffix_id = 1     
        self.partition_queue = PartitionQueueFile(filepath_prefix + PARTITION_QUEUE_NAME)
        self.cd_patterns_left = 0        
        self._heap_dirty = False         
        self._not_output_mask = None

    def __pattern_row_mask(self, pattern_idx) -> bitarray:
        combined = None
        for idx in pattern_idx:
            m = self.item_row_masks[self.F1[idx]['name']]
            combined = m.copy() if combined is None else (combined & m)
        return combined

    def __sync_not_output_mask(self):
        if self.output_row_mask_np is None:
            self._not_output_mask = None
            return
        num_rows = len(self.dataset)
        if self.output_row_mask_np.shape[0] != num_rows:
            self._not_output_mask = None
            return
        self._not_output_mask = self.np_packed_not(self.output_row_mask_np, num_rows)

    @property
    def high_potential_phase_ended(self) -> bool:
        if self.pattern_registry is not None and self.pattern_registry.cusum_monitor is not None:
            return self.pattern_registry.cusum_monitor.high_potential_phase_ended
        return False


    def __item_token_length(self, item_idx: int) -> int:
        name = self.F1[item_idx]['name']
        tl = self.item_token_len_cache.get(name)
        if tl is None:
            tl = len(self.tokenizer.encode(name.split(": _", 1)[1]))
            self.item_token_len_cache[name] = tl
        return tl

    def __write_node_patterns(self, N: FrequentItemsetTreeNode,
                              include_own: bool, include_absorbed: bool) -> int:
        n_count = N.cur_count if N.cur_count is not None else N.count
        
        written = 0
        itemset_list = []  
        base = [self.F1[i]['name'] for i in N.pattern_idx]
        if include_own:
            itemset_list.append((base, "own"))
            self.num_of_frequent_itemsets += 1
        if include_absorbed:
            absorbed = N.inherited_absorbed + tuple(N.own_absorbed or ())
            num_combos = 1 << len(absorbed)
            for i in range(1, num_combos):
                itemset = list(base)
                for b in range(len(absorbed)):
                    if (i & (1 << b)) > 0:
                        itemset.append(self.F1[absorbed[b]]['name'])
                cols = [self.__item_column(nm) for nm in itemset]
                if len(cols) != len(set(cols)):
                    continue
                itemset_list.append((itemset, "absorbed"))
                self.num_of_frequent_itemsets += 1
        for itemset, kind in itemset_list:
            level = len(itemset)
            while len(self.filename_list) < level:
                assert len(self.filename_list) == len(self.writer_list)
                next_level = len(self.filename_list) + 1
                self.filename_list.append(self.filepath_prefix + "_mining_F" + str(next_level) + ".jsonl")
                self.writer_list.append(open(self.filename_list[next_level - 1], 'w'))
            write_file_with_handle(self.writer_list[level - 1], [itemset],
                                   self.tokenizer, [n_count], self.pattern_registry,
                                   kind=kind)
            written += 1
        return written


    def __deduplicate_rows(self):
        seen = {}
        dedup_dataset = []
        dup_row_map = []   
        for orig_idx, row in enumerate(self.dataset):
            row_key = tuple(row)
            first_idx = seen.get(row_key)
            if first_idx is not None:
                dup_row_map[first_idx].append(orig_idx)   
            else:
                seen[row_key] = len(dup_row_map)
                dup_row_map.append([orig_idx])
                dedup_dataset.append(row)
        num_orig = len(self.dataset)
        self.dataset = dedup_dataset
        self.dup_row_map = dup_row_map
        del seen   
        num_dup = num_orig - len(dedup_dataset)


    def __find_F1(self):
        item_name_to_count = {}
        self.num_of_transactions = 0
        for transaction in self.dataset:
            self.num_of_transactions += 1
            for item_name in transaction:
                item_count = item_name_to_count.setdefault(item_name, 0)
                item_name_to_count[item_name] = item_count + 1
        item_name_to_count.pop('', None)
        original_F1_count = len([cnt for cnt in item_name_to_count.values() if cnt >= self.min_count])
        self.min_count = 2
        self.F1 = [{'name': item_name, 'count': item_count} for (item_name, item_count) in item_name_to_count.items() if self.min_count <= item_count]
        for item in self.F1:
            item['token_length'] = len(self.tokenizer.encode(item['name'].split(": _", 1)[1]))
            item['potential'] = item['token_length'] * (item['count'] - 1) / item['count']
        self.F1.sort(key=lambda item: (-item['potential'], -item['token_length'], -item['count']))

        self.__build_item_row_masks()

        self.F1 = tuple(self.F1)

    def __generate_NodeSets_of_1_itemsets(self):
        item_name_to_item_index = {item['name']: item_index for (item_index, item) in enumerate(self.F1)}
        self.item_to_NodeSet = {item_index: [] for item_index in item_name_to_item_index.values()}
        bmc_tree_root = BMCTreeNode(item=None, count=None, bitmap_code=bitarray([False] * len(self.F1)))
        for transaction in self.dataset:
            transaction = [item_name_to_item_index[item_name] for item_name in transaction if item_name in item_name_to_item_index]
            transaction.sort(reverse=True)
            cur_root = bmc_tree_root
            for item in transaction:
                N = cur_root.get_child_registering_item(item)
                if N is None:
                    bitmap_code = cur_root.bitmap_code.copy()
                    bitmap_code[item] = True
                    N = BMCTreeNode(item=item, count=0, bitmap_code=bitmap_code)
                    cur_root.add_child(N)
                    self.item_to_NodeSet[item].append(N)
                N.count += 1
                cur_root = N
        clean_BMC_tree(bmc_tree_root)

    def __expand_node(self, N: FrequentItemsetTreeNode):
        N.children = []
        N.own_absorbed = []
        is_level1 = (len(N.pattern_idx) == 1)

        used_cols = {self.__item_column(self.F1[i]['name']) for i in N.pattern_idx}
        used_cols.update(self.__item_column(self.F1[i]['name']) for i in N.inherited_absorbed)

        not_output = self._not_output_mask
        parent_remaining_mask = None
        if not_output is not None:
            parent_remaining_mask = self.__pattern_row_mask(N.pattern_idx)
            if parent_remaining_mask is not None:
                parent_remaining_mask &= not_output

        for j in range(N.sib_start, len(N.siblings)):
            S = N.siblings[j]
            s_col = self.__item_column(self.F1[S.item]['name'])
            if s_col in used_cols:
                continue
            child = FrequentItemsetTreeNode()
            sum_of_NegNodeSets_counts = 0
            if is_level1:
                for ni in N.NegNodeSet:
                    if not ni.bitmap_code[S.item]:
                        child.NegNodeSet.append(ni)
                        sum_of_NegNodeSets_counts += ni.count
            else:
                for nj in S.NegNodeSet:
                    if nj.bitmap_code[N.item]:
                        child.NegNodeSet.append(nj)
                        sum_of_NegNodeSets_counts += nj.count
            child.count = N.count - sum_of_NegNodeSets_counts

            if self.min_count <= child.count:
                if N.count == child.count:
                    N.own_absorbed.append(S.item)
                    used_cols.add(s_col)   
                else:
                    child.item = S.item
                    child.parent = N
                    child.pattern_idx = N.pattern_idx + (S.item,)
                    child.token_length = N.token_length + self.__item_token_length(S.item)

                    if parent_remaining_mask is not None:
                        s_name = self.F1[S.item]['name']
                        child_remaining = (parent_remaining_mask
                                           & self.item_row_masks[s_name]).count()
                        child.cur_count = child_remaining
                        if child_remaining <= 0:
                            continue
                        if child_remaining == 1:
                            child.potential = 0.0   
                        else:
                            child.potential = (child.token_length
                                               * (child_remaining - 1) / child_remaining)
                    else:
                        child.cur_count = None
                        child.potential = (child.token_length
                                           * (child.count - 1) / child.count)

                    child.siblings = N.children
                    child.sib_start = len(N.children) + 1
                    N.children.append(child)

        inherited = N.inherited_absorbed + tuple(N.own_absorbed) if N.own_absorbed else N.inherited_absorbed
        for C in N.children:
            C.inherited_absorbed = inherited


    def __create_root_of_frequent_itemset_tree(self):
        root = FrequentItemsetTreeNode()
        for item in range(len(self.F1)):
            child = FrequentItemsetTreeNode()
            child.item = item
            child.count = self.F1[item]['count']
            child.NegNodeSet = self.item_to_NodeSet[item]
            root.children.append(child)
        return root

    def runAlgorithm(self):
        start_timestamp = datetime.now()

        self.__find_F1()                              
        self.__generate_NodeSets_of_1_itemsets()
        root = self.__create_root_of_frequent_itemset_tree()

        heap = []
        self._tie_counter = itertools.count()
        root_children = root.children
        num_f1 = len(root_children)
        for i, R in enumerate(root_children):
            R.pattern_idx = (R.item,)
            R.siblings = root_children
            R.sib_start = i + 1
            R.inherited_absorbed = ()
            R.token_length = self.__item_token_length(R.item)
            R.potential = R.token_length * (R.count - 1) / R.count   
            heapq.heappush(heap, (-R.potential, -R.token_length, -R.count,
                                  next(self._tie_counter), R))

        pop_count = 0
        while heap:
            if self._heap_dirty:
                self.__refresh_heap_after_partition(heap)
                self._heap_dirty = False

            _, _, _, _, N = heapq.heappop(heap)       
            self.__expand_node(N)                        
            n_written = self.__write_node_patterns(N, include_own=True, include_absorbed=True)
            for C in N.children:
                if C.count < self.min_count:  
                    continue
                c_key = C.cur_count if C.cur_count is not None else C.count
                heapq.heappush(heap, (-C.potential, -C.token_length, -c_key,
                                      next(self._tie_counter), C))

            N.NegNodeSet = []
            N.siblings = None
            N.parent = None
            N.own_absorbed = None

            pop_count += 1
            if pop_count % 1000 == 0:
                pass

            self.__after_patterns_written(n_written)

        monitor = self.pattern_registry.cusum_monitor if self.pattern_registry else None
        if monitor is not None:
            if monitor.high_potential_phase_ended:
                pass
            elif monitor.mu0 is None:
                pass
            else:
                pass

        num_rows = len(self.dataset)
        if self.output_row_mask_np is None or self.output_row_mask_np.shape[0] != num_rows:
            self.output_row_mask_np = np.zeros(num_rows, dtype=bool)
        remaining = np.flatnonzero(~self.output_row_mask_np)

        final_name = os.path.basename(
            f"{self.filepath_prefix}_partition_{self.partition_suffix_id}.txt")
        with open(f"{self.filepath_prefix}_partition_{self.partition_suffix_id}.txt",
                  'w', encoding='utf-8') as f:
            for rid in self.__expand_to_original_ids(remaining):
                f.write(f"{rid}\n")
            f.flush()
            os.fsync(f.fileno())   
        self.partition_suffix_id += 1

        self.partition_queue.push_unconditional(final_name)   

        write_mining_end_signal(self.filepath_prefix)
        for writter in self.writer_list:
            writter.close()
        self.execution_time = (datetime.now() - start_timestamp).total_seconds() * 1000

    def __build_item_row_masks(self):
        num_rows = len(self.dataset)
        self.item_row_masks = {f1_item['name']: bitarray(num_rows) for f1_item in self.F1}
        for row_idx, row in enumerate(self.dataset):
            for item_name in row:
                mask = self.item_row_masks.get(item_name)
                if mask is not None:
                    mask[row_idx] = 1
        self.pattern_registry = PatternRegistry(item_masks=self.item_row_masks, num_rows=num_rows)
        self.pattern_registry.on_high_potential_ended = self._on_high_potential_ended
        self.__write_initial_uncovered_partition()


    def __finalize_pattern_containment(self):
        registry = self.pattern_registry.key_to_node
        for node in registry.values():
            node.sub_patterns = []
            for it in node.items:
                sub_node = registry.get(node.item_key - frozenset([it]))
                if sub_node is not None:
                    node.sub_patterns.append(sub_node)

    def compute_pattern_score(self, node: PatternNode) -> float:
        def dfs(current_node: PatternNode, remaining_depth: int) -> float:
            if remaining_depth <= 0 or not current_node.sub_patterns:
                return 0.0
            share_count = current_node.unextracted_count
            if share_count <= 0:
                return 0.0
            best_gain = 0.0
            for child in current_node.sub_patterns:
                step_gain = child.token_length / share_count                
                branch_gain = step_gain + dfs(child, remaining_depth - 1)   
                if branch_gain > best_gain:
                    best_gain = branch_gain
            return best_gain

        return node.token_length + dfs(node, SUB_PATTERN_DEPTH_L - 1)

    @staticmethod
    def np_packed_not(mask_np: np.ndarray, num_rows: int) -> bitarray:
        inverted = (~mask_np).astype(np.uint8)
        packed = np.packbits(inverted, bitorder='big')
        ba = bitarray(endian='big')
        ba.frombytes(packed.tobytes())
        return ba[:num_rows]

    def _on_high_potential_ended(self):
        self.__extract_and_write_partition()
        self.cd_patterns_left = PARTITION_CD

    def __write_initial_uncovered_partition(self):
        num_rows = len(self.dataset)
        self.output_row_mask_np = np.zeros(num_rows, dtype=bool)

        covered = np.zeros(num_rows, dtype=bool)
        for mask in self.item_row_masks.values():
            buf = np.frombuffer(mask.tobytes(), dtype=np.uint8)
            bits = np.unpackbits(buf, bitorder='big')[:num_rows]
            covered |= bits.astype(bool)

        uncovered = np.flatnonzero(~covered)
        if uncovered.size == 0:
            return

        self.output_row_mask_np[uncovered] = True   
        orig_ids = self.__expand_to_original_ids(uncovered)

        partition_name = os.path.basename(
            f"{self.filepath_prefix}_partition_{self.partition_suffix_id}.txt")
        with open(f"{self.filepath_prefix}_partition_{self.partition_suffix_id}.txt",
                  'w', encoding='utf-8') as f:
            for rid in orig_ids:
                f.write(f"{rid}\n")
            f.flush()
            os.fsync(f.fileno())
        self.partition_suffix_id += 1

        ok = self.partition_queue.push(partition_name)   
        self._heap_dirty = True   

    def __extract_and_write_partition(self):
        if self.pattern_registry is None or not self.pattern_registry.key_to_node:
            return

        num_rows = len(self.dataset)
        if self.output_row_mask_np is None or self.output_row_mask_np.shape[0] != num_rows:
            self.output_row_mask_np = np.zeros(num_rows, dtype=bool)

        self.__finalize_pattern_containment()
        all_nodes = list(self.pattern_registry.key_to_node.values())
        for node in all_nodes:
            node.score = self.compute_pattern_score(node)
        all_nodes.sort(key=lambda n: n.score, reverse=True)
        all_nodes = [n for n in all_nodes if n.unextracted_count > 0]

        partition_rows = []
        newly_output_count = 0
        for node in all_nodes:
            if node.unextracted_rows is not None:
                rows = node.unextracted_rows
                rows_to_add = rows[~self.output_row_mask_np[rows]]
            elif node.unextracted_mask is not None:
                buf = np.frombuffer(node.unextracted_mask.tobytes(), dtype=np.uint8)
                bits = np.unpackbits(buf, bitorder='big')[:num_rows]
                rows = np.flatnonzero(bits)
                rows_to_add = rows[~self.output_row_mask_np[rows]]
            else:
                continue
            if rows_to_add.size == 0:
                continue

            if newly_output_count + rows_to_add.size <= ROW_SCALE_THRESHOLD_S:
                self.output_row_mask_np[rows_to_add] = True
                partition_rows.append(rows_to_add)
                newly_output_count += int(rows_to_add.size)
                node.unextracted_rows = None
                node.unextracted_mask = None
                node.unextracted_count = 0
                if newly_output_count == ROW_SCALE_THRESHOLD_S:
                    break   
            else:
                need = ROW_SCALE_THRESHOLD_S - newly_output_count
                chosen, remaining = self.__hard_threshold_select(
                    node, rows_to_add, need, all_nodes)
                if chosen.size:
                    self.output_row_mask_np[chosen] = True
                    partition_rows.append(chosen)
                    newly_output_count += int(chosen.size)   

                if remaining.size:
                    node.unextracted_rows = remaining.astype(np.int32)
                    node.unextracted_mask = None
                    node.unextracted_count = int(remaining.size)
                else:
                    node.unextracted_rows = None
                    node.unextracted_mask = None
                    node.unextracted_count = 0
                break   



        not_packed = self.np_packed_not(self.output_row_mask_np, num_rows)
        for node in all_nodes:
            if node.unextracted_count > 0:
                if node.unextracted_rows is not None:
                    keep = ~self.output_row_mask_np[node.unextracted_rows]
                    node.unextracted_rows = node.unextracted_rows[keep]
                    node.unextracted_count = int(node.unextracted_rows.size)
                elif node.unextracted_mask is not None:
                    node.unextracted_mask &= not_packed
                    node.unextracted_count = node.unextracted_mask.count()

        if not partition_rows:
            return

        dedup_ids = np.concatenate(partition_rows)
        orig_ids = self.__expand_to_original_ids(dedup_ids)

        partition_name = os.path.basename(
            f"{self.filepath_prefix}_partition_{self.partition_suffix_id}.txt")
        with open(f"{self.filepath_prefix}_partition_{self.partition_suffix_id}.txt",
                  'w', encoding='utf-8') as f:
            for rid in orig_ids:
                f.write(f"{rid}\n")
            f.flush()
            os.fsync(f.fileno())
        self.partition_suffix_id += 1

        ok = self.partition_queue.push(partition_name)
        self.__sync_not_output_mask()
        self._heap_dirty = True

    def __hard_threshold_select(self, pattern_a: PatternNode,
                                candidate_rows: np.ndarray, need: int,
                                all_nodes: list) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(candidate_rows, dtype=np.int64)
        n = rows.size
        num_rows = len(self.dataset)

        pos_arr = np.full(num_rows, -1, dtype=np.int64)
        pos_arr[rows] = np.arange(n)
        cand_mask = np.zeros(num_rows, dtype=bool)
        cand_mask[rows] = True
        score_lists = [[] for _ in range(n)]
        for node in all_nodes:
            if node is pattern_a:        
                continue
            if node.unextracted_rows is not None:
                p_rows = node.unextracted_rows
            elif node.unextracted_mask is not None:
                buf = np.frombuffer(node.unextracted_mask.tobytes(), dtype=np.uint8)
                bits = np.unpackbits(buf, bitorder='big')[:num_rows]
                p_rows = np.flatnonzero(bits)
            else:
                continue
            inter = p_rows[cand_mask[p_rows]]
            if inter.size == 0:
                continue
            s = node.score
            for r in inter.tolist():
                score_lists[pos_arr[r]].append(s)

        max_len = max((len(ls) for ls in score_lists), default=0)
        keys = []
        for i in range(n):
            ls = sorted(score_lists[i], reverse=True) 
            neg = [-x for x in ls]                     
            neg.extend([float('inf')] * (max_len - len(ls))) 
            keys.append(tuple(neg) + (int(rows[i]),))   
        order = sorted(range(n), key=lambda i: keys[i])

        chosen_idx = np.asarray(order[:need], dtype=np.int64)
        chosen = rows[chosen_idx]
        remain_mask = np.ones(n, dtype=bool)
        remain_mask[chosen_idx] = False
        return chosen, rows[remain_mask]


    def __expand_to_original_ids(self, dedup_ids: np.ndarray) -> list:
        if self.dup_row_map is not None:
            orig = []
            for did in dedup_ids.tolist():
                orig.extend(self.dup_row_map[did])
            orig.sort()
            return orig
        return dedup_ids.tolist()

    def __after_patterns_written(self, n_new: int):
        if not self.high_potential_phase_ended:
            return  
        for _ in range(n_new):
            if self.cd_patterns_left > 0:
                self.cd_patterns_left -= 1
                continue
            if self.partition_queue.pending_count() < PARTITION_QUEUE_CAPACITY:
                self.__extract_and_write_partition()
                self.cd_patterns_left = PARTITION_CD

    def __refresh_heap_after_partition(self, heap: list):
        if not heap:
            return
        not_output = self._not_output_mask
        if not_output is None:
            return
        refreshed = []
        dropped = 0
        for _, _, _, _, node in heap:
            combined = self.__pattern_row_mask(node.pattern_idx)
            if combined is None:
                new_count = node.count
            else:
                combined &= not_output
                new_count = combined.count()
            if new_count <= 0:
                dropped += 1
                continue
            node.cur_count = new_count
            if new_count == 1:
                node.potential = 0.0
            else:
                node.potential = node.token_length * (new_count - 1) / new_count
            refreshed.append((-node.potential, -node.token_length, -new_count,
                              next(self._tie_counter), node))
        heapq.heapify(refreshed)
        heap[:] = refreshed  


    def extract_high_value_rows(self):
        start_timestamp = datetime.now()
        if self.pattern_registry is None or not self.pattern_registry.key_to_node:
            return

        all_nodes = list(self.pattern_registry.key_to_node.values())
        self.__finalize_pattern_containment()

        for node in all_nodes:
            node.score = self.compute_pattern_score(node)

        all_nodes.sort(key=lambda n: n.score, reverse=True)

        num_rows = len(self.dataset)
        if self.output_row_mask_np is None or self.output_row_mask_np.shape[0] != num_rows:
            self.output_row_mask_np = np.zeros(num_rows, dtype=bool)

        prev_count = int(self.output_row_mask_np.sum())
        newly_output_count = 0   

        for node in all_nodes:
            if node.unextracted_rows is not None:
                rows = node.unextracted_rows
                rows_to_add = rows[~self.output_row_mask_np[rows]]
            elif node.unextracted_mask is not None:
                buf  = np.frombuffer(node.unextracted_mask.tobytes(), dtype=np.uint8)
                bits = np.unpackbits(buf, bitorder='big')[:num_rows]
                rows = np.flatnonzero(bits)
                rows_to_add = rows[~self.output_row_mask_np[rows]]
            else:
                rows_to_add = np.empty(0, dtype=np.int64)

            if rows_to_add.size:
                self.output_row_mask_np[rows_to_add] = True   
                newly_output_count += int(rows_to_add.size)

            node.unextracted_rows = None
            node.unextracted_mask = None
            node.unextracted_count = 0

            if prev_count + newly_output_count >= ROW_SCALE_THRESHOLD_S:
                break

        total_output = prev_count + newly_output_count

        not_packed = self.np_packed_not(self.output_row_mask_np, num_rows)
        for node in all_nodes:
            if node.unextracted_count > 0:
                if node.unextracted_rows is not None:
                    keep = ~self.output_row_mask_np[node.unextracted_rows]
                    node.unextracted_rows = node.unextracted_rows[keep]
                    node.unextracted_count = int(node.unextracted_rows.size)
                elif node.unextracted_mask is not None:
                    node.unextracted_mask &= not_packed
                    node.unextracted_count = node.unextracted_mask.count()

        end_timestamp = datetime.now()
        elapsed_ms = (end_timestamp - start_timestamp).total_seconds() * 1000
        self.extract_time_records.append(elapsed_ms)

        expanded_total = len(self.get_selected_original_row_ids(expand_duplicates=True))


    def get_selected_original_row_ids(self, expand_duplicates: bool = True) -> list:
        if self.output_row_mask_np is None or not self.output_row_mask_np.any():
            return []
        if expand_duplicates and self.dup_row_map is not None:
            orig_ids = []
            for dedup_id in np.flatnonzero(self.output_row_mask_np).tolist():
                orig_ids.extend(self.dup_row_map[dedup_id])
            orig_ids.sort()
            return orig_ids
        return np.flatnonzero(self.output_row_mask_np).tolist()

    @staticmethod
    def __item_column(item_name: str) -> str:
        return item_name.split(": _", 1)[0]


    def get_selected_rows_dataframe(self, expand_duplicates: bool = True) -> pd.DataFrame:
        orig_ids = self.get_selected_original_row_ids(expand_duplicates)
        if not orig_ids:
            return pd.DataFrame()
        return self.source_df.iloc[orig_ids]

def process_pattern_mining(
    dataset_path: str,
    output_prefix: str,
    skip_alpha: float,
    llm_path: str,
    input_type: str,
    task_id: int,
    columns_list: list[str]
):
    dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
    dataset = dataset[columns_list]
    removed_count = 0
    for f in glob.glob(output_prefix + "*"):
        if os.path.isfile(f):
            os.remove(f)
        removed_count += 1
    
    base_dir = os.path.dirname(os.path.abspath(output_prefix)) or "."
    removed_partitions = 0
    for f in glob.glob(os.path.join(base_dir, "*_partition_*.txt")):
        if os.path.isfile(f):
            os.remove(f)
            removed_partitions += 1

    mining = NegFIN(dataset, output_prefix, skip_alpha, llm_path, task_id)
    mining.runAlgorithm()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config/tree_eval.ini', encoding='utf-8')
    dataset_path = config.get('backend', 'dataset_path')
    output_prefix = config.get('backend', 'output_prefix')
    columns_name = config.get('backend', 'columns_name')
    columns_list = [col.strip() for col in columns_name.split(',')]
    skip_alpha = config.getfloat('backend', 'skip_alpha')
    llm_path = config.get('backend', 'llm_path')
    dataset_type = config.get('backend', 'dataset_type')
    process_pattern_mining(dataset_path, output_prefix, skip_alpha, llm_path, dataset_type, 1, columns_list)
