"""Portable sparse MLA baseline for the V4.1 token-major cache formats.

The cache stays packed in HBM. Only the selected rows are dequantized inside
the attention kernel. All byte offsets are computed in int64, including when
a sparse token index fits in int32 but its cache byte offset does not.

Unlike the CUDA split-KV implementation, this baseline does not cache a
length-dependent worklist: topk lengths are read from device memory on every
launch, including graph replay.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(code):
    magnitude = code & 7
    value = tl.where(
        magnitude < 2,
        magnitude * 0.5,
        tl.where(
            magnitude < 4,
            1.0 + (magnitude - 2) * 0.5,
            tl.where(magnitude < 6, 2.0 + (magnitude - 4), 4.0 + (magnitude - 6) * 2.0),
        ),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _attend(
    q,
    cache,
    indices,
    lengths,
    row,
    heads,
    m,
    l,
    acc,
    Q_PER_BATCH: tl.constexpr,
    N_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    N_PAGES: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    TOPK: tl.constexpr,
    FORMAT: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch, seq = row // Q_PER_BATCH, row % Q_PER_BATCH
    length = TOPK
    if HAS_LENGTH:
        length = tl.minimum(tl.maximum(tl.load(lengths + batch), 0), TOPK)
    d = tl.arange(0, 512)
    for start in range(tl.cdiv(length, BLOCK_K)):
        j = start * BLOCK_K + tl.arange(0, BLOCK_K)
        token = tl.load(indices + batch * IS0 + seq * IS1 + j * IS2, j < length, -1).to(
            tl.int64
        )
        valid = (j < length) & (token >= 0) & (token < N_PAGES * PAGE_SIZE)
        base = (token // PAGE_SIZE) * CS0 + (token % PAGE_SIZE) * CS1
        if FORMAT == 0:
            kv = tl.load(cache + base[:, None] + d[None, :], valid[:, None], 0).to(
                tl.bfloat16
            )
        elif FORMAT == 8:
            data = tl.load(cache + base[:, None] + d[None, :], valid[:, None], 0)
            scale_bits = tl.load(
                cache + base[:, None] + 512 + d[None, :] // 32, valid[:, None], 127
            )
            # UE8M0 has no zero or infinity: code 0 is 2^-127 and code
            # 255 is NaN, unlike the exponent field of an IEEE FP32 value.
            bits = scale_bits.to(tl.uint32) << 23
            bits = tl.where(scale_bits == 0, 0x00400000, bits)
            bits = tl.where(scale_bits == 255, 0x7FC00000, bits)
            scale = bits.to(tl.float32, bitcast=True)
            kv = (data.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale).to(
                tl.bfloat16
            )
        else:
            packed = tl.load(cache + base[:, None] + d[None, :] // 2, valid[:, None], 0)
            code = (packed >> ((d[None, :] & 1) * 4)) & 15
            scale_bits = tl.load(
                cache + base[:, None] + 256 + d[None, :] // 16, valid[:, None], 0
            )
            scale = scale_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            kv = (_e2m1(code) * scale).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(kv)) * SM_SCALE
        scores = tl.where(
            valid[None, :] & (heads[:, None] < N_HEADS), scores, -float("inf")
        )
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(scores - safe_m[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), kv, acc)
        m = new_m
    return m, l, acc


@triton.jit
def _sparse_mla(
    Q,
    C,
    I,
    L,
    E,
    EI,
    EL,
    SINK,
    O,
    MAX,
    LSE,
    Q_PER_BATCH: tl.constexpr,
    N_HEADS: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    QS3: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    N_PAGES: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    TOPK: tl.constexpr,
    FORMAT: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    EXTRA: tl.constexpr,
    E_PAGE_SIZE: tl.constexpr,
    E_N_PAGES: tl.constexpr,
    ES0: tl.constexpr,
    ES1: tl.constexpr,
    EIS0: tl.constexpr,
    EIS1: tl.constexpr,
    EIS2: tl.constexpr,
    E_TOPK: tl.constexpr,
    E_FORMAT: tl.constexpr,
    E_HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SINK_STRIDE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, 512)
    q = tl.load(
        Q
        + (row // Q_PER_BATCH) * QS0
        + (row % Q_PER_BATCH) * QS1
        + heads[:, None] * QS2
        + d[None, :] * QS3,
        heads[:, None] < N_HEADS,
        0,
    )
    m = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_H,), 0.0, tl.float32)
    acc = tl.full((BLOCK_H, 512), 0.0, tl.float32)
    m, l, acc = _attend(
        q,
        C,
        I,
        L,
        row,
        heads,
        m,
        l,
        acc,
        Q_PER_BATCH,
        N_HEADS,
        PAGE_SIZE,
        N_PAGES,
        CS0,
        CS1,
        IS0,
        IS1,
        IS2,
        TOPK,
        FORMAT,
        HAS_LENGTH,
        SM_SCALE,
        BLOCK_K,
    )
    if EXTRA:
        m, l, acc = _attend(
            q,
            E,
            EI,
            EL,
            row,
            heads,
            m,
            l,
            acc,
            Q_PER_BATCH,
            N_HEADS,
            E_PAGE_SIZE,
            E_N_PAGES,
            ES0,
            ES1,
            EIS0,
            EIS1,
            EIS2,
            E_TOPK,
            E_FORMAT,
            E_HAS_LENGTH,
            SM_SCALE,
            BLOCK_K,
        )
    safe_l = tl.where(l > 0, l, 1.0)
    lse = tl.where(l > 0, m + tl.log(safe_l), float("inf"))
    output = acc / safe_l[:, None]
    if HAS_SINK:
        sink = tl.load(SINK + heads * SINK_STRIDE, heads < N_HEADS, 0)
        sink_weight = tl.where(l > 0, 1.0 / (1.0 + tl.exp(sink - lse)), 0.0)
        output = output * sink_weight[:, None]
    tl.store(
        O + row.to(tl.int64) * N_HEADS * 512 + heads[:, None] * 512 + d[None, :],
        output,
        heads[:, None] < N_HEADS,
    )
    tl.store(MAX + row * N_HEADS + heads, m, heads < N_HEADS)
    tl.store(LSE + row * N_HEADS + heads, lse, heads < N_HEADS)


def _validate_scope(q, cache, indices, lengths, *, allow_bf16=False):
    if cache.device != q.device or indices.device != q.device:
        raise ValueError("query, cache and indices must be on the same device")
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.stride(-1) != 1:
        raise ValueError("expected a token-major [pages, page_size, 1, row] MQA cache")
    if (
        indices.shape[:2] != q.shape[:2]
        or indices.ndim != 3
        or indices.dtype != torch.int32
    ):
        raise ValueError("indices must be int32 [batch, sequence, topk]")
    if lengths is not None and (
        lengths.shape != (q.shape[0],)
        or lengths.dtype != torch.int32
        or lengths.device != q.device
    ):
        raise ValueError("topk lengths must be int32 [batch] on the query device")
    if allow_bf16 and cache.shape[-1] == 512 and cache.dtype == torch.bfloat16:
        return cache, 0
    if cache.shape[-1] not in (528, 288) or cache.element_size() != 1:
        raise ValueError(
            "V4.1 cache rows must contain 528 FP8 bytes or 288 packed FP4 bytes"
        )
    return cache.view(torch.uint8), 8 if cache.shape[-1] == 528 else 4


def _forward(
    q,
    cache,
    indices,
    lengths,
    scale,
    sink,
    *,
    extra=None,
    extra_indices=None,
    extra_lengths=None,
    prefill=False,
):
    if q.ndim != 4 or q.shape[-1] != 512 or q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError(
            "query must be a BF16 GPU tensor [batch, sequence, heads, 512]"
        )
    if sink is not None and (
        sink.shape != (q.shape[2],)
        or sink.device != q.device
        or sink.dtype != torch.float32
    ):
        raise ValueError("attention sinks must be float32 [heads] on the query device")
    cache, fmt = _validate_scope(q, cache, indices, lengths, allow_bf16=prefill)
    if (extra is None) != (extra_indices is None):
        raise ValueError("extra cache and indices must be supplied together")
    if extra is None and extra_lengths is not None:
        raise ValueError("extra lengths require an extra cache")
    has_extra = extra is not None
    if has_extra:
        extra, efmt = _validate_scope(q, extra, extra_indices, extra_lengths)
    else:
        extra, extra_indices, efmt = cache, indices, fmt
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    maxima = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    lse = torch.empty_like(maxima)
    _sparse_mla[(q.shape[0] * q.shape[1], triton.cdiv(q.shape[2], 16))](
        q,
        cache,
        indices,
        lengths,
        extra,
        extra_indices,
        extra_lengths,
        sink,
        out,
        maxima,
        lse,
        q.shape[1],
        q.shape[2],
        *q.stride(),
        cache.shape[1],
        cache.shape[0],
        cache.stride(0),
        cache.stride(1),
        *indices.stride(),
        indices.shape[-1],
        fmt,
        lengths is not None,
        has_extra,
        extra.shape[1],
        extra.shape[0],
        extra.stride(0),
        extra.stride(1),
        *extra_indices.stride(),
        extra_indices.shape[-1],
        efmt,
        extra_lengths is not None,
        sink is not None,
        sink.stride(0) if sink is not None else 1,
        scale,
        16,
        32,
        num_warps=4,
    )
    return out, maxima, lse


def sparse_prefill_fwd(q, kv, indices, sm_scale, d_v, attn_sink, topk_length):
    if d_v != 512 or kv.ndim != 3 or kv.shape[1:] != (1, 512):
        raise ValueError("V4.1 prefill requires MQA with d_qk=d_v=512")
    out, maxima, lse = _forward(
        q.unsqueeze(1),
        kv.unsqueeze(1),
        indices,
        topk_length,
        sm_scale,
        attn_sink,
        prefill=True,
    )
    return out.squeeze(1), maxima.squeeze(1), lse.squeeze(1)


def sparse_decode_fwd(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    tile_scheduler_metadata,
    num_splits,
    extra_k_cache,
    extra_indices,
    extra_topk_length,
    head_dim_v,
    softmax_scale,
    enable_batch_invariant,
):
    if head_dim_v != 512 or k_cache.shape[-1] != 528:
        raise ValueError("V4.1 decode requires a 528-byte FP8 main cache and d_v=512")
    out, _, lse = _forward(
        q,
        k_cache,
        indices,
        topk_length,
        softmax_scale,
        attn_sink,
        extra=extra_k_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_topk_length,
    )
    # This implementation has no split worklist. Empty tensors explicitly encode
    # that, rather than retaining stale metadata from a previous batch length.
    if tile_scheduler_metadata is None:
        tile_scheduler_metadata = torch.empty(0, dtype=torch.int32, device=q.device)
    if num_splits is None:
        num_splits = torch.empty(0, dtype=torch.int32, device=q.device)
    return out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits
