import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_fasttree_decode_stage1(
    Q, K, V,
    stride_q0, stride_q1,
    stride_k0, stride_k1,
    stride_v0, stride_v1,
    stride_o0, stride_o1,
    stride_lse0,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    node_offset,
    sm_scale,
    MidO,
    MidLSE,
    head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    KV_TILE_SIZE: tl.constexpr,
    stride_r2t_0,
    vnode_to_kv_req_indices,
    vnode_to_kv_token_offsets,
    vnode_to_mask,
    vnode_to_mask_offs,
     vnode_to_mask_valid,   
    HAS_MASK: tl.constexpr,   
):
    cur_vnode = tl.program_id(0) + node_offset
    cur_kv_head = tl.program_id(1)

    Q_BLOCK_SIZE: tl.constexpr = Q_TILE_SIZE * kv_group_num

    offs_d = tl.arange(0, head_dim)
    offs_g = tl.arange(0, kv_group_num)

    cur_q_entry_start = tl.load(vnode_to_q_offs + cur_vnode)
    cur_q_len = tl.load(vnode_to_q_lens + cur_vnode)
    cur_kv_entry_start = tl.load(vnode_to_kv_offs + cur_vnode)
    cur_kv_len = tl.load(vnode_to_kv_lens + cur_vnode)
    cur_mask_off = tl.load(vnode_to_mask_offs + cur_vnode * HAS_MASK)
    cur_mask_valid_i = tl.load(vnode_to_mask_valid + cur_vnode * HAS_MASK)  
    cur_mask_valid = cur_mask_valid_i.to(tl.float32)   

    cur_qo_head = cur_kv_head * kv_group_num

    offs_q_block = tl.arange(0, Q_BLOCK_SIZE)
    mask_row = offs_q_block // kv_group_num
    mask_row = tl.where(mask_row < cur_q_len, mask_row, 0) * HAS_MASK
    mask_row = mask_row * cur_mask_valid_i

    offs_q_tile = tl.arange(0, Q_TILE_SIZE)
    q_len_mask = (offs_q_tile < cur_q_len).to(tl.int32)
    offs_q_tile_safe = offs_q_tile * q_len_mask
    q_entries_raw = tl.load(
        vnode_to_q_entries + cur_q_entry_start + offs_q_tile_safe
    )
    q_entries = q_entries_raw * q_len_mask + (q_len_mask - 1)
    q_entries_mask = (q_entries >= 0).to(tl.int32)
    q_entries_safe = q_entries * q_entries_mask

    off_q = (
        q_entries_safe[:, None, None] * stride_q0
        + (cur_qo_head + offs_g[None, :, None]) * stride_q1
        + offs_d[None, None, :]
    )
    q = tl.load(Q + off_q)
    q = (q.to(tl.float32) * (q_entries >= 0)[:, None, None]).to(q.type)
    q = q.reshape([Q_BLOCK_SIZE, head_dim])

    acc = tl.zeros([Q_BLOCK_SIZE, head_dim], dtype=tl.float32)
    sum_exp = tl.zeros([Q_BLOCK_SIZE, 1], dtype=tl.float32)
    max_logics = tl.full([Q_BLOCK_SIZE, 1], float("-inf"), dtype=tl.float32)

    offs_kv_tile = tl.arange(0, KV_TILE_SIZE)
    for kv_tile_start in range(0, cur_kv_len, KV_TILE_SIZE):
        offs_kv_tile_new = offs_kv_tile + kv_tile_start
        valid_kv_mask = (offs_kv_tile_new < cur_kv_len).to(tl.int32)
        offs_kv_tile_new_safe = offs_kv_tile_new * valid_kv_mask

        kv_req_idx = tl.load(
            vnode_to_kv_req_indices + cur_kv_entry_start + offs_kv_tile_new_safe
        )
        kv_tok_off = tl.load(
            vnode_to_kv_token_offsets + cur_kv_entry_start + offs_kv_tile_new_safe
        )
        kv_flat_offset = kv_req_idx * stride_r2t_0 + kv_tok_off
        kv_entries_raw = tl.load(vnode_to_kv_entries + kv_flat_offset)
        kv_entries = kv_entries_raw * valid_kv_mask + (valid_kv_mask - 1)
        kv_entries_safe_mask = (kv_entries >= 0).to(tl.int32)
        kv_entries_safe = kv_entries * kv_entries_safe_mask

        off_k = (
            kv_entries_safe[None, :] * stride_k0
            + cur_kv_head * stride_k1
            + offs_d[:, None]
        )
        k = tl.load(K + off_k)
        k = (k.to(tl.float32) * (kv_entries >= 0)[None, :]).to(k.type)

        if KV_TILE_SIZE >= 16 and Q_BLOCK_SIZE >= 16:
            qk = tl.dot(q, k)
        else:
            qk = tl.sum(
                q[:, :, None].to(tl.float32) * k[None, :, :].to(tl.float32),
                axis=1,
            )
        qk *= sm_scale

        valid_kv_final = ((valid_kv_mask == 1) & (kv_entries >= 0)).to(tl.float32)

        mask_off = (
            cur_mask_off
            + mask_row[:, None] * cur_kv_len
            + offs_kv_tile_new_safe[None, :] * cur_mask_valid_i
        )
        raw_keep = tl.load(vnode_to_mask + mask_off * HAS_MASK).to(tl.float32)
        qkv_keep = raw_keep * cur_mask_valid * HAS_MASK + (1.0 - cur_mask_valid * HAS_MASK)
        valid = valid_kv_final[None, :] * qkv_keep

        qk = qk * valid + (1.0 - valid) * (-1.0e6)

        off_v = (
            kv_entries_safe[:, None] * stride_v0
            + cur_kv_head * stride_v1
            + offs_d[None, :]
        )
        v = tl.load(V + off_v)
        v = (v.to(tl.float32) * (kv_entries >= 0)[:, None]).to(v.type)

        cur_max_logics = tl.max(qk, axis=1)[:, None]
        new_max_logics = tl.maximum(cur_max_logics, max_logics)
        exp_logics = tl.exp(qk - new_max_logics)
        exp_logics = exp_logics * valid
        logic_scale = tl.exp(max_logics - new_max_logics)
        sum_exp = sum_exp * logic_scale + tl.sum(exp_logics, axis=1)[:, None]
        acc = acc * logic_scale
        if KV_TILE_SIZE >= 16 and Q_BLOCK_SIZE >= 16:
            acc = tl.dot(exp_logics.to(v.type), v, acc)
        else:
            acc = acc + tl.sum(
                exp_logics[:, :, None] * v[None, :, :].to(tl.float32), axis=1
            )
        max_logics = new_max_logics

    off_mid_o = (
        (cur_q_entry_start + offs_q_tile[:, None, None]) * stride_o0
        + (cur_qo_head + offs_g[None, :, None]) * stride_o1
        + offs_d[None, None, :]
    )
    off_mid_lse = (
        (cur_q_entry_start + offs_q_tile[:, None]) * stride_lse0
        + cur_qo_head
        + offs_g[None, :]
    )

    sum_exp_safe = tl.where(sum_exp > 0, sum_exp, 1.0)
    mid_o = (acc / sum_exp_safe).reshape([Q_TILE_SIZE, kv_group_num, head_dim])
    q_store_mask = q_entries[:, None, None] >= 0
    tl.store(
        MidO + off_mid_o,
        mid_o,
        mask=q_store_mask,
    )

    mid_lse = (tl.log(sum_exp) + max_logics).reshape([Q_TILE_SIZE, kv_group_num])
    q_store_mask_lse = q_entries[:, None] >= 0
    tl.store(
        MidLSE + off_mid_lse,
        mid_lse,
        mask=q_store_mask_lse,
    )


def _fasttree_decode_stage1(
    Q, K, V,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    sm_scale,
    MidO,
    MidLSE,
    phase_node_nums,
    phase_node_offsets,
    head_num,
    head_dim,
    kv_head_num,
    phase_q_tile_sizes,
    phase_kv_tile_sizes,
    vnode_to_kv_req_indices,
    vnode_to_kv_token_offsets,
    vnode_to_mask,        
    vnode_to_mask_offs,   
    vnode_to_mask_valid
):
    Lq, Lk = Q.shape[-1], K.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256}
    kv_group_num = head_num // kv_head_num
    stride_r2t_0 = vnode_to_kv_entries.stride(0)

    has_mask = vnode_to_mask is not None  

    for node_num, node_offset, Q_TILE_SIZE, KV_TILE_SIZE in zip(
        phase_node_nums, phase_node_offsets, phase_q_tile_sizes, phase_kv_tile_sizes,
    ):
        if node_num == 0:
            continue
        grid = (node_num, kv_head_num)
        _fwd_fasttree_decode_stage1[grid](
            Q,
            K,
            V,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            V.stride(0), V.stride(1),
            MidO.stride(0), MidO.stride(1),
            MidLSE.stride(0),
            vnode_to_kv_entries,
            vnode_to_kv_offs,
            vnode_to_kv_lens,
            vnode_to_q_entries,
            vnode_to_q_offs,
            vnode_to_q_lens,
            node_offset,
            sm_scale,
            MidO,
            MidLSE,
            head_dim,
            kv_group_num,
            Q_TILE_SIZE,
            KV_TILE_SIZE,
            stride_r2t_0,
            vnode_to_kv_req_indices,
            vnode_to_kv_token_offsets,
            vnode_to_mask,        
            vnode_to_mask_offs,   
            vnode_to_mask_valid,   
            int(has_mask),   
        )


@triton.jit
def _fwd_fasttree_decoding_stage2(
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    MidO,
    MidLSE,
    O,
    stride_mid_o0,
    stride_mid_o1,
    stride_mid_lse0,
    stride_o0,
    stride_o1,
    head_dim: tl.constexpr,
    NODE_TILE_SIZE: tl.constexpr,
):
    cur_req = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_vnode_entry_start = tl.load(req_to_vnode_offs + cur_req)
    cur_vnode_len = tl.load(req_to_vnode_lens + cur_req)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    max_lse = tl.full([1], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([1, head_dim], dtype=tl.float32)
    offs_d = tl.arange(0, head_dim)
    offs_n = tl.arange(0, NODE_TILE_SIZE)

    for tile_start in range(0, cur_vnode_len, NODE_TILE_SIZE):
        offs_n_new = offs_n + tile_start
        vnode_entries = tl.load(
            req_to_vnode_entries + cur_vnode_entry_start + offs_n_new
        )
        vnode_entries = tl.where(offs_n_new < cur_vnode_len, vnode_entries, -1)

        off_mid_o = (
            vnode_entries[:, None] * stride_mid_o0
            + cur_head * stride_mid_o1
            + offs_d[None, :]
        )
        mid_o = tl.load(
            MidO + off_mid_o, mask=vnode_entries[:, None] >= 0, other=0.0
        )
        off_mid_lse = vnode_entries * stride_mid_lse0 + cur_head
        mid_lse = tl.load(
            MidLSE + off_mid_lse, mask=vnode_entries >= 0, other=float("-inf")
        )[None, :]

        tile_max_lse = tl.max(mid_lse, axis=1)
        new_max_lse = tl.maximum(tile_max_lse, max_lse)
        exp_lse = tl.exp(mid_lse - new_max_lse)
        old_scale = tl.exp(max_lse - new_max_lse)
        sum_exp = sum_exp * old_scale + tl.sum(exp_lse, axis=1)
        acc = acc * old_scale + tl.sum(exp_lse[:, :, None] * mid_o[None, :, :], axis=1)
        max_lse = new_max_lse

    tl.store(
        O + cur_req * stride_o0 + cur_head * stride_o1 + offs_d,
        tl.ravel(acc / sum_exp).to(O.type.element_ty),
    )


def _fasttree_decode_stage2(
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    MidO,
    MidLSE,
    O,
    batch_size,
    head_num,
    head_dim,
):
    NODE_TILE_SIZE = 4
    grid = (batch_size, head_num)
    _fwd_fasttree_decoding_stage2[grid](
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        MidO,
        MidLSE,
        O,
        MidO.stride(0),
        MidO.stride(1),
        MidLSE.stride(0),
        O.stride(0),
        O.stride(1),
        head_dim,
        NODE_TILE_SIZE,
    )


def fasttree_decode(
    q,
    k_buffer,
    v_buffer,
    o,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    mid_o,
    mid_lse,
    vnode_to_mask,           
    vnode_to_mask_offs,      
    vnode_to_mask_valid,
    phase_node_nums,
    phase_node_offsets,
    phase_q_tile_sizes,
    phase_kv_tile_sizes,
    sm_scale,
    logit_cap=-1,
    vnode_to_kv_req_indices=None,
    vnode_to_kv_token_offsets=None,
):
    head_num = q.shape[1]
    head_dim = q.shape[-1]
    kv_head_num = k_buffer.shape[1]

    if vnode_to_mask is None:
        vnode_to_mask = torch.empty(1, dtype=torch.int8, device=q.device)
        vnode_to_mask_offs = torch.empty(1, dtype=torch.int32, device=q.device)
        vnode_to_mask_valid = torch.zeros(1, dtype=torch.int8, device=q.device)

    _fasttree_decode_stage1(
        q,
        k_buffer,
        v_buffer,
        vnode_to_kv_entries,
        vnode_to_kv_offs,
        vnode_to_kv_lens,
        vnode_to_q_entries,
        vnode_to_q_offs,
        vnode_to_q_lens,
        sm_scale,
        mid_o,
        mid_lse,
        phase_node_nums,
        phase_node_offsets,
        head_num,
        head_dim,
        kv_head_num,
        phase_q_tile_sizes,
        phase_kv_tile_sizes,
        vnode_to_kv_req_indices,
        vnode_to_kv_token_offsets,
        vnode_to_mask,
        vnode_to_mask_offs,
        vnode_to_mask_valid,   
    )
    batch_size = q.shape[0]
    _fasttree_decode_stage2(
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        mid_o,
        mid_lse,
        o,
        batch_size,
        head_num,
        head_dim,
    )
