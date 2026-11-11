import torch
import queue
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from fasttree.metadata import FastTreeMetadata

_RA = 4
_RS = 16
_ALPHA = 0.8
_BETA = 0.2
_MAX_EXTRA_DECODE_STEPS = 128

_SOURCE_COUNTER = {"next": 0}

def _next_source_id():
    _SOURCE_COUNTER["next"] += 1
    return _SOURCE_COUNTER["next"]

def _make_bin(lv: dict) -> dict:
    sid = frozenset((_next_source_id(),))
    return {
        "q_entries": list(lv["q_entries"]),
        "kv_req_indices": list(lv["kv_req_indices"]),
        "kv_token_offsets": list(lv["kv_token_offsets"]),
        "q_sources": [set(sid) for _ in lv["q_entries"]],
        "kv_sources": [set(sid) for _ in lv["kv_req_indices"]],
    }

def _absorb_bin(b: dict, lv: dict) -> None:
    q_pos = {q: i for i, q in enumerate(b["q_entries"])}
    for q, srcs in zip(lv["q_entries"], lv["q_sources"]):
        i = q_pos.get(q)
        if i is None:
            q_pos[q] = len(b["q_entries"])
            b["q_entries"].append(q)
            b["q_sources"].append(set(srcs))
        else:
            b["q_sources"][i] |= set(srcs)
    kv_pos = {(r, t): i for i, (r, t)
              in enumerate(zip(b["kv_req_indices"], b["kv_token_offsets"]))}
    for (r, t), srcs in zip(
        zip(lv["kv_req_indices"], lv["kv_token_offsets"]), lv["kv_sources"]
    ):
        i = kv_pos.get((r, t))
        if i is None:
            kv_pos[(r, t)] = len(b["kv_req_indices"])
            b["kv_req_indices"].append(r)
            b["kv_token_offsets"].append(t)
            b["kv_sources"].append(set(srcs))
        else:
            b["kv_sources"][i] |= set(srcs)

def _combine_bins(items: List[dict]) -> dict:
    q_entries, q_sources = [], []
    kv_req, kv_tok, kv_sources = [], [], []
    q_pos, kv_pos = {}, {}
    for bd in items:
        for q, srcs in zip(bd["q_entries"], bd["q_sources"]):
            i = q_pos.get(q)
            if i is None:
                q_pos[q] = len(q_entries)
                q_entries.append(q)
                q_sources.append(set(srcs))
            else:
                q_sources[i] |= set(srcs)
        for (r, t), srcs in zip(
            zip(bd["kv_req_indices"], bd["kv_token_offsets"]), bd["kv_sources"]
        ):
            key = (r, t)
            i = kv_pos.get(key)
            if i is None:
                kv_pos[key] = len(kv_req)
                kv_req.append(r)
                kv_tok.append(t)
                kv_sources.append(set(srcs))
            else:
                kv_sources[i] |= set(srcs)
    return {
        "q_entries": q_entries,
        "kv_req_indices": kv_req,
        "kv_token_offsets": kv_tok,
        "q_sources": q_sources,
        "kv_sources": kv_sources,
    }

def _build_bin_mask(b: dict):
    if not b["q_entries"] or not b["kv_req_indices"]:
        return None
    all_srcs = set()
    for s in b["q_sources"]:
        all_srcs |= s
    for s in b["kv_sources"]:
        all_srcs |= s
    if len(all_srcs) <= 1:
        return None
    return [
        [1 if (qs & ks) else 0 for ks in b["kv_sources"]]
        for qs in b["q_sources"]
    ]


def prepare_for_fasttree_meta(
    batch: ScheduleBatch,
    fasttree_metadata: FastTreeMetadata,
):
    assert isinstance(batch.tree_cache, RadixCache)
    root_node = batch.tree_cache.root_node

    node_to_reqs: Dict[TreeNode, List[int]] = defaultdict(list)
    for rid, req in enumerate(batch.reqs):
        node = root_node
        prefix_len = len(req.origin_input_ids)
        output_index = -prefix_len
        while output_index < len(req.output_ids):
            token_index = output_index + prefix_len
            if token_index >= prefix_len:
                break
            token = (
                req.output_ids[output_index]
                if output_index >= 0
                else req.origin_input_ids[token_index]
            )
            child = node.children.get(token)
            if child:
                node_to_reqs[child].append(rid)
                node = child
                output_index += len(child.key)
            else:
                break

    q_tile_size = fasttree_metadata.TSQs[0]
    kv_tile_size = fasttree_metadata.TSKs[0]
    kv_split_size = fasttree_metadata.kv_split_sizes[0]

    nodes_in_bfs_order: List[TreeNode] = []
    node_token_offsets: List[int] = []
    node_parent: Dict[TreeNode, Optional[TreeNode]] = {root_node: None}
    node_que: queue.Queue[Tuple[TreeNode, int]] = queue.Queue()
    node_que.put((root_node, 0))

    while not node_que.empty():
        node, offset = node_que.get()
        nodes_in_bfs_order.append(node)
        node_token_offsets.append(offset)
        child_offset = offset + len(node.value)
        for child in node.children.values():
            if child in node_to_reqs.keys():
                node_parent[child] = node
                node_que.put((child, child_offset))

    node_to_bfs_idx: Dict[TreeNode, int] = {
        node: i for i, node in enumerate(nodes_in_bfs_order)
    }

    def traverse():
        vnode_num_aggr_reqs = [len(node_to_reqs[node]) for node in nodes_in_bfs_order]
        vnode_seqlens = [len(node.value) for node in nodes_in_bfs_order]
        node_assignments = [0 for _ in vnode_seqlens]
        node_kv_start_offsets = list(node_token_offsets)

        for i, node in enumerate(nodes_in_bfs_order):
            nQcurr = vnode_num_aggr_reqs[i]
            lenv = vnode_seqlens[i]
            for child in node.children.values():
                if child not in node_to_reqs.keys():
                    continue
                assert child in node_to_bfs_idx.keys(), "Child node not found in BFS index map"
                l = node_to_bfs_idx[child]
                nQl = len(node_to_reqs[child])
                lenl = vnode_seqlens[l]
                parent_len = lenv
                child_nQ = nQl
                if 2 * parent_len - 3 * child_nQ >= 0:
                    node_assignments[l] = 0
                else:
                    node_assignments[l] = 1
                    nQcurr -= nQl
                    vnode_seqlens[l] = lenl + lenv
                    node_kv_start_offsets[l] = node_kv_start_offsets[i]
            vnode_num_aggr_reqs[i] = nQcurr
            
        return node_assignments, vnode_num_aggr_reqs, vnode_seqlens, node_kv_start_offsets

    node_assignments, vnode_num_aggr_reqs, vnode_seqlens, node_kv_start_offsets = traverse()

    logical_vnodes: List[dict] = []
    node_to_lv_idx: Dict[TreeNode, int] = {}

    for i, (node, num_aggr_reqs, seqlen, token_offset) in enumerate(
        zip(
            nodes_in_bfs_order,
            vnode_num_aggr_reqs,
            vnode_seqlens,
            node_kv_start_offsets,
        )
    ):
        if num_aggr_reqs <= 0:
            continue

        merged_child_reqs: set = set()
        for child in node.children.values():
            if child not in node_to_reqs.keys():
                continue
            child_idx = node_to_bfs_idx[child]
            if node_assignments[child_idx] == 1:
                merged_child_reqs.update(node_to_reqs[child])

        q_entries: List[int] = [
            rid for rid in node_to_reqs[node]
            if rid not in merged_child_reqs
        ]

        first_req = q_entries[0] if len(q_entries) > 0 else node_to_reqs[node][0]

        kv_req_indices: List[int] = [first_req] * seqlen
        kv_token_offsets: List[int] = list(range(token_offset, token_offset + seqlen))
        assert len(kv_req_indices) == len(kv_token_offsets)

        lv = {
            "q_entries": q_entries,
            "kv_req_indices": kv_req_indices,
            "kv_token_offsets": kv_token_offsets,
            "tree_node": node,
        }
        sid = frozenset((_next_source_id(),))
        lv["q_sources"] = [set(sid) for _ in q_entries]
        lv["kv_sources"] = [set(sid) for _ in kv_req_indices]
        node_to_lv_idx[node] = len(logical_vnodes)
        logical_vnodes.append(lv)

    bins: List[dict] = []
    lv_to_bin: Dict[int, int] = {}
    merged = [False] * len(logical_vnodes)

    for lv_idx, lv in enumerate(logical_vnodes):
        if merged[lv_idx]:
            continue

        q_size = len(lv["q_entries"])
        kv_size = len(lv["kv_req_indices"])
        q_tile = q_tile_size
        kv_tile = kv_tile_size

        if q_size > q_tile or kv_size > kv_tile:
            bins.append(_make_bin(lv))
            lv_to_bin[lv_idx] = len(bins) - 1
            merged[lv_idx] = True
            continue

        candidate_bin_indices: List[int] = []
        tree_node = lv["tree_node"]
        parent_node = node_parent.get(tree_node)
        cur_idx = node_to_bfs_idx[tree_node]
            
        n_ancestors = 0
        ancestor = parent_node
        while ancestor is not None and n_ancestors < _RA:
            n_ancestors += 1                      
            anc_bfs = node_to_bfs_idx.get(ancestor)
            anc_lv_idx = node_to_lv_idx.get(ancestor)
            if anc_bfs is not None and anc_lv_idx is not None and anc_lv_idx in lv_to_bin:
                candidate_bin_indices.append(lv_to_bin[anc_lv_idx])
            if anc_bfs is None or node_assignments[anc_bfs] == 0:
                break                             
            ancestor = node_parent.get(ancestor)
            
        if parent_node is not None:
            children_list = list(parent_node.children.values())
            try:
                current_pos = children_list.index(tree_node)
            except ValueError:
                current_pos = -1
            if current_pos > 0:
                window_start = max(0, current_pos - _RS)   
                for pos in range(current_pos - 1, window_start - 1, -1):
                    sibling = children_list[pos]
                    if sibling not in node_to_lv_idx:
                        continue
                    sibling_bfs = node_to_bfs_idx.get(sibling)
                    if sibling_bfs is None or node_assignments[sibling_bfs] != 1:
                        continue
                    sibling_lv_idx = node_to_lv_idx[sibling]
                    if sibling_lv_idx in lv_to_bin:
                        candidate_bin_indices.append(lv_to_bin[sibling_lv_idx])
        n_levels = 0
        child_on_path = tree_node
        ancestor = parent_node
        found_split = False
        while ancestor is not None and n_levels < _RA:
            n_levels += 1
            if not found_split and node_assignments[node_to_bfs_idx[child_on_path]] == 0:
                found_split = True
            if found_split:
                anc_lv_idx = node_to_lv_idx.get(ancestor)
                if anc_lv_idx is not None and anc_lv_idx in lv_to_bin:
                    candidate_bin_indices.append(lv_to_bin[anc_lv_idx])
            child_on_path = ancestor
            ancestor = node_parent.get(ancestor)

        candidate_bin_indices = list(dict.fromkeys(candidate_bin_indices))

        lv_q_set = set(lv["q_entries"])
        lv_kv_set = set(zip(lv["kv_req_indices"], lv["kv_token_offsets"]))

        best_bin_idx = -1
        best_benefit = 0

        for bin_idx in candidate_bin_indices:
            b = bins[bin_idx]

            b_q_set = set(b["q_entries"])
            b_kv_set = set(zip(b["kv_req_indices"], b["kv_token_offsets"]))

            combined_q_size = len(b_q_set | lv_q_set)
            combined_kv_size = len(b_kv_set | lv_kv_set)

            if combined_q_size > q_tile or combined_kv_size > kv_tile:
                continue

            q_overlap = len(lv_q_set & b_q_set)
            kv_overlap = len(lv_kv_set & b_kv_set)

            if q_overlap == 0 and kv_overlap == 0:
                continue

            # if q_overlap > 0:
            #     memory = 3 * q_overlap
            # else:
            #     memory = 2 * kv_overlap
            memory = 3 * q_overlap + 2 * kv_overlap

            q_excl_lv = lv_q_set - b_q_set
            q_excl_bin = b_q_set - lv_q_set
            kv_excl_lv = lv_kv_set - b_kv_set
            kv_excl_bin = b_kv_set - lv_kv_set
            padding = len(kv_excl_bin) * len(q_excl_lv) + len(kv_excl_lv) * len(q_excl_bin)

            benefit = _ALPHA * memory - _BETA * padding

            if benefit > best_benefit:
                best_benefit = benefit
                best_bin_idx = bin_idx

        if best_bin_idx >= 0:
            b = bins[best_bin_idx]
            _absorb_bin(b, lv)
            lv_to_bin[lv_idx] = best_bin_idx
            merged[lv_idx] = True
        else:
            bins.append(_make_bin(lv))
            lv_to_bin[lv_idx] = len(bins) - 1
            merged[lv_idx] = True

    target_q_tile = q_tile_size
    target_kv_tile = kv_tile_size

    fillable_bins: List[dict] = []
    skip_bins: List[dict] = []
    for bin_d in bins:
        if (
            len(bin_d["q_entries"]) > target_q_tile
            or len(bin_d["kv_req_indices"]) > target_kv_tile
        ):
            skip_bins.append(bin_d)
        else:
            fillable_bins.append(bin_d)

    n_fillable = len(fillable_bins)
    if n_fillable > 0:
        b = sum(len(bd["q_entries"]) for bd in fillable_bins) / n_fillable
        a = sum(len(bd["kv_req_indices"]) for bd in fillable_bins) / n_fillable
    else:
        a = 0.0
        b = 0.0

    new_bins_p2: List[dict] = []
    new_bins_p2.extend(skip_bins)

    if n_fillable > 0:
        score_buckets: Dict[Tuple[int, int], List[dict]] = {}
        for bin_d in fillable_bins:
            key = (len(bin_d["q_entries"]), len(bin_d["kv_req_indices"]))
            if key not in score_buckets:
                score_buckets[key] = []
            score_buckets[key].append(bin_d)

        bucket_keys = sorted(
            score_buckets.keys(),
            key=lambda qk: (-(a * qk[0] + b * qk[1]), -qk[0], -qk[1]),
        )

        containers: List[dict] = []
        for key in bucket_keys:
            for bin_d in score_buckets[key]:
                q_size, kv_size = key
                placed = False
                if containers:
                    last_c = containers[-1]
                    if (
                        last_c["q_used"] + q_size <= target_q_tile
                        and last_c["kv_used"] + kv_size <= target_kv_tile
                    ):
                        last_c["q_used"] += q_size
                        last_c["kv_used"] += kv_size
                        last_c["items"].append(bin_d)
                        placed = True
                if not placed:
                    containers.append(
                        {
                            "q_used": q_size,
                            "kv_used": kv_size,
                            "items": [bin_d],
                        }
                    )

        for c in containers:
            if len(c["items"]) == 1:
                new_bins_p2.append(c["items"][0])
                continue
            new_bins_p2.append(_combine_bins(c["items"]))

    bins = new_bins_p2


    vnode_to_kv_offs: List[int] = []
    vnode_to_kv_lens: List[int] = []
    vnode_to_q_entries: List[int] = []
    vnode_to_q_offs: List[int] = []
    vnode_to_q_lens: List[int] = []
    vnode_to_kv_req_indices: List[int] = []
    vnode_to_kv_token_offsets: List[int] = []
    vnode_to_mask: List[int] = []        
    vnode_to_mask_offs: List[int] = []   
    vnode_to_mask_valid: List[int] = []  

    for b in bins:
        q_entries = b["q_entries"]
        kv_req_indices = b["kv_req_indices"]
        kv_token_offsets = b["kv_token_offsets"]
        bin_mask = _build_bin_mask(b)    
        assert len(kv_req_indices) == len(kv_token_offsets), "KV req indices and token offsets must be the same length"


        q_size = len(q_entries)
        kv_size = len(kv_req_indices)
        q_split_size = q_tile_size
        q_split_count = (q_size - 1) // q_split_size + 1
        kv_split_count = (kv_size - 1) // kv_split_size + 1

        for kv_split_id in range(kv_split_count):
            q_offset_start = len(vnode_to_q_entries)
            vnode_to_q_entries.extend(q_entries)

            split_kv_off = kv_split_id * kv_split_size
            vnode_kv_len = min(split_kv_off + kv_split_size, kv_size) - split_kv_off

            kv_offset_start = len(vnode_to_kv_req_indices)
            vnode_to_kv_req_indices.extend(
                kv_req_indices[split_kv_off : split_kv_off + vnode_kv_len]
            )
            vnode_to_kv_token_offsets.extend(
                kv_token_offsets[split_kv_off : split_kv_off + vnode_kv_len]
            )

            for q_split_id in range(q_split_count):
                split_q_off = q_split_id * q_split_size
                vnode_q_len = min(split_q_off + q_split_size, q_size) - split_q_off
                
                if bin_mask is None:
                    vnode_to_mask_valid.append(0)
                    vnode_to_mask_offs.append(0)      
                else:
                    mask_off = len(vnode_to_mask)
                    for qi in range(split_q_off, split_q_off + vnode_q_len):
                        vnode_to_mask.extend(
                            bin_mask[qi][split_kv_off : split_kv_off + vnode_kv_len]
                        )
                    vnode_to_mask_valid.append(1)
                    vnode_to_mask_offs.append(mask_off)

                vnode_to_kv_offs.append(kv_offset_start)
                vnode_to_kv_lens.append(vnode_kv_len)
                vnode_to_q_offs.append(q_offset_start + split_q_off)
                vnode_to_q_lens.append(vnode_q_len)

    req_last_vnodes = [0 for _ in range(len(batch.reqs))]

    for rid, req in enumerate(batch.reqs):
        prefix_len = len(req.origin_input_ids)
        generated_len = len(req.output_ids) + 1
        for kv_split_id in range((generated_len - 1) // kv_split_size + 1):
            q_offset = len(vnode_to_q_entries)
            vnode_to_q_entries.append(rid)
            vnode_to_q_offs.append(q_offset)
            vnode_to_q_lens.append(1)
            split_kv_off = kv_split_id * kv_split_size
            vnode_kv_len = (
                min(split_kv_off + kv_split_size, generated_len)
                - split_kv_off
            )
            kv_offset_start = len(vnode_to_kv_req_indices)
            is_last_split = split_kv_off + kv_split_size >= generated_len
            extra = _MAX_EXTRA_DECODE_STEPS if is_last_split else 0
            vnode_to_kv_req_indices.extend([rid] * (vnode_kv_len + extra))
            vnode_to_kv_token_offsets.extend(
                range(
                    prefix_len + split_kv_off,
                    prefix_len + split_kv_off + vnode_kv_len + extra,
                )
            )
            vnode_to_mask_valid.append(0)
            vnode_to_mask_offs.append(0)
            vnode_to_kv_offs.append(kv_offset_start)
            vnode_to_kv_lens.append(vnode_kv_len)

        req_last_vnodes[rid] = len(vnode_to_kv_lens) - 1

    fasttree_metadata.req_last_vnodes = req_last_vnodes
    fasttree_metadata.phase_q_tile_sizes = [q_tile_size] * 4
    fasttree_metadata.phase_kv_tile_sizes = [kv_tile_size] * 4

    req_to_vnode_entries: List[List[int]] = [[] for _ in range(len(batch.reqs))]
    req_to_vnode_offs: List[int] = []
    req_to_vnode_lens: List[int] = []

    for i, req in enumerate(vnode_to_q_entries):
        req_to_vnode_entries[req].append(i)

    offset = 0
    for i in range(len(batch.reqs)):
        req_to_vnode_offs.append(offset)
        offset += len(req_to_vnode_entries[i])
        req_to_vnode_lens.append(len(req_to_vnode_entries[i]))
    req_to_vnode_entries = [
        item for sublist in req_to_vnode_entries for item in sublist
    ]

    fasttree_metadata.phase_node_nums = [len(vnode_to_q_lens), 0, 0, 0]
    fasttree_metadata.phase_node_offsets = [0, 0, 0, 0]

    def list_to_gpu_tensor(preallocated_gpu, cpu_list, dtype=torch.int32):
        cpu_tensor = torch.tensor(cpu_list, device="cpu", dtype=dtype)
        n = cpu_tensor.size(0)
        if n > 0:
            preallocated_gpu[:n].copy_(cpu_tensor, non_blocking=True)

    list_to_gpu_tensor(fasttree_metadata.vnode_to_q_entries, vnode_to_q_entries)
    list_to_gpu_tensor(fasttree_metadata.vnode_to_q_offs, vnode_to_q_offs)
    list_to_gpu_tensor(fasttree_metadata.vnode_to_q_lens, vnode_to_q_lens)
    list_to_gpu_tensor(fasttree_metadata.vnode_to_kv_offs, vnode_to_kv_offs)
    list_to_gpu_tensor(fasttree_metadata.vnode_to_kv_lens, vnode_to_kv_lens)
    list_to_gpu_tensor(
        fasttree_metadata.vnode_to_kv_req_indices, vnode_to_kv_req_indices
    )
    list_to_gpu_tensor(
        fasttree_metadata.vnode_to_kv_token_offsets, vnode_to_kv_token_offsets
    )
    list_to_gpu_tensor(fasttree_metadata.req_to_vnode_entries, req_to_vnode_entries)
    list_to_gpu_tensor(fasttree_metadata.req_to_vnode_offs, req_to_vnode_offs)
    list_to_gpu_tensor(fasttree_metadata.req_to_vnode_lens, req_to_vnode_lens)
    fasttree_metadata.ensure_mask_capacity(len(vnode_to_mask))   
    list_to_gpu_tensor(fasttree_metadata.vnode_to_mask, vnode_to_mask, dtype=torch.int8)      
    list_to_gpu_tensor(fasttree_metadata.vnode_to_mask_offs, vnode_to_mask_offs)              
    list_to_gpu_tensor(fasttree_metadata.vnode_to_mask_valid, vnode_to_mask_valid)  
    batch.fasttree_metadata = fasttree_metadata


def prepare_for_fasttree_step(
    batch: ScheduleBatch,
    fasttree_metadata: FastTreeMetadata,
):
    bs = len(batch.reqs)
    increment = torch.ones(bs, device=fasttree_metadata.device, dtype=torch.int32)
    fasttree_metadata.vnode_to_kv_lens[fasttree_metadata.req_last_vnodes] += increment
