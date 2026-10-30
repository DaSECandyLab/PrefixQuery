import torch
from typing import Tuple, List


class FastTreeMetadata:
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        TSQs: Tuple[int, int, int, int] = None, 
        TSKs: Tuple[int, int, int, int] = (32, 32, 16, 16),
        kv_split_sizes: Tuple[int, int] = (1024, 128), 
        para_threshs1: Tuple[int, int, int, int] = (132, 132, 132, 132),
        para_threshs2: Tuple[int, int, int, int] = (132, 528, 132, 528),
        max_vnodes: int = 4096,
        max_vnode_to_q_entries: int = 24576,
        max_vnode_to_kv_entries: int = 49152,
        max_batch_size: int = 4096,
        q_thresh: int = 32,
        kv_thresh: int = 64,
        max_vnode_to_mask_entries: int = None,   
    ):
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_group_num = num_qo_heads // num_kv_heads
        self.device = device
        
        self.TSQs = TSQs if TSQs is not None else (
            64 // self.kv_group_num, 16 // self.kv_group_num,
            64 // self.kv_group_num, 16 // self.kv_group_num
        )
        self.TSKs = TSKs
        self.kv_split_sizes = kv_split_sizes
        self.para_threshs1 = para_threshs1
        self.para_threshs2 = para_threshs2
        self.q_thresh = q_thresh
        self.kv_thresh = kv_thresh
        
        if max_vnode_to_mask_entries is None:
            max_vnode_to_mask_entries = 32 * 1024 * 1024   
        self.max_vnode_to_mask_entries = max_vnode_to_mask_entries


        with torch.device(device):
            self.vnode_to_q_entries = torch.empty(
                max_vnode_to_q_entries, dtype=torch.int32
            )
            self.vnode_to_q_offs = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_q_lens = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_kv_offs = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_kv_lens = torch.empty(max_vnodes, dtype=torch.int32)

            self.vnode_to_kv_req_indices = torch.empty(
                max_vnode_to_kv_entries, dtype=torch.int32
            )
            self.vnode_to_kv_token_offsets = torch.empty(
                max_vnode_to_kv_entries, dtype=torch.int32
            )

            self.req_to_vnode_entries = torch.empty(
                max_vnode_to_q_entries, dtype=torch.int32
            )
            self.req_to_vnode_offs = torch.empty(max_batch_size, dtype=torch.int32)
            self.req_to_vnode_lens = torch.empty(max_batch_size, dtype=torch.int32)
            self.mid_o = torch.empty(
                (max_vnode_to_q_entries, num_qo_heads, head_dim), dtype=torch.float32
            )
            self.mid_lse = torch.empty(
                (max_vnode_to_q_entries, num_qo_heads), dtype=torch.float32
            )
            
            self.vnode_to_mask = torch.empty(
                max_vnode_to_mask_entries, dtype=torch.int8
            )
            self.vnode_to_mask_valid = torch.empty(max_vnodes, dtype=torch.int8)
            self.vnode_to_mask_offs = torch.empty(max_vnodes, dtype=torch.int32)

        self.phase_node_nums: Tuple[int, int, int, int] = None
        self.phase_node_offsets: Tuple[int, int, int, int] = None
        self.phase_q_tile_sizes: Tuple[int, int, int, int] = None
        self.phase_kv_tile_sizes: Tuple[int, int, int, int] = None

        self.req_last_vnodes: List[int] = None
    
    def ensure_mask_capacity(self, need: int):
        if need <= self.max_vnode_to_mask_entries:
            return
        new_cap = max(need, self.max_vnode_to_mask_entries * 2)
        self.vnode_to_mask = torch.empty(
            new_cap, dtype=torch.int8, device=self.device
        )
        self.max_vnode_to_mask_entries = new_cap
