"""ROCm correctness tests using the public V4.1 quantization contract."""

import pytest
import torch

from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCm"
)


def _pack(x, bits):
    n, page, _, d = x.shape
    tile = 32 if bits == 8 else 16
    z = x.float().reshape(n, page, d // tile, tile)
    amax = z.abs().amax(-1)
    if bits == 8:
        scale = torch.exp2((amax / 448).clamp_min(1e-4).log2().ceil())
        data = (z / scale[..., None]).to(torch.float8_e4m3fn)
        scale_bytes = (scale.log2() + 127).to(torch.uint8)
        decoded = data.float() * scale[..., None]
        packed = data.reshape(n, page, d).view(torch.uint8)
    else:
        scale = (amax / 6).clamp(2**-9, 448).to(torch.float8_e4m3fn)
        z = z / scale.float()[..., None]
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
        mids = (table[:-1] + table[1:]) / 2
        lower = torch.bucketize(z.abs().contiguous(), mids, right=False)
        upper = torch.bucketize(z.abs().contiguous(), mids, right=True)
        # At a midpoint choose the code with an even mantissa LSB.
        codes = torch.where(lower != upper, lower + (lower & 1), lower).to(torch.uint8)
        decoded = (
            table[codes.long()]
            * torch.where(torch.signbit(z), -1.0, 1.0)
            * scale.float()[..., None]
        )
        codes = (codes | (torch.signbit(z).to(torch.uint8) << 3)).reshape(n, page, d)
        packed = codes[..., ::2] | (codes[..., 1::2] << 4)
        scale_bytes = scale.view(torch.uint8)
    return torch.cat([packed, scale_bytes], dim=-1).unsqueeze(2), decoded.reshape(
        n, page, 1, d
    ).bfloat16()


def _reference(q, scopes, sink):
    rows, h, d = q.shape[0] * q.shape[1], q.shape[2], q.shape[3]
    values, valid = [], []
    for kv, indices, lengths in scopes:
        flat = kv.reshape(-1, d).float()
        mask = (indices >= 0) & (indices < len(flat))
        if lengths is not None:
            mask &= (
                torch.arange(indices.shape[-1], device=q.device)[None, None, :]
                < lengths[:, None, None]
            )
        values.append(flat[indices.clamp(0, len(flat) - 1).long()].reshape(rows, -1, d))
        valid.append(mask.reshape(rows, -1))
    kv, mask = torch.cat(values, 1), torch.cat(valid, 1)
    if kv.shape[1] == 0:
        return (
            torch.zeros_like(q),
            torch.full(q.shape[:-1], -torch.inf, device=q.device),
            torch.full(q.shape[:-1], torch.inf, device=q.device),
        )
    logits = torch.bmm(q.reshape(rows, h, d).float(), kv.transpose(1, 2)) * (d**-0.5)
    logits.masked_fill_(~mask[:, None, :], -torch.inf)
    lse, maximum = logits.logsumexp(-1), logits.amax(-1)
    norm = lse if sink is None else torch.logaddexp(lse, sink)
    weights = torch.exp(
        logits - norm.masked_fill(norm == -torch.inf, torch.inf)[..., None]
    )
    out = torch.bmm(weights, kv).reshape(q.shape).bfloat16()
    lse.masked_fill_(lse == -torch.inf, torch.inf)
    return out, maximum.reshape(q.shape[:-1]), lse.reshape(q.shape[:-1])


@pytest.mark.parametrize("topk", [0, 1, 31, 32, 33, 63, 64, 65, 129])
def test_prefill_lengths_and_masking(topk):
    torch.manual_seed(11)
    q = torch.randn(3, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(257, 1, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(-2, 259, (3, 1, topk), device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [0, max(0, topk - 1), topk], device="cuda", dtype=torch.int32
    )
    sink = torch.linspace(-3, 3, 64, device="cuda")
    out, maximum, lse = flash_mla_sparse_fwd(
        q, kv, indices, 512**-0.5, attn_sink=sink, topk_length=lengths
    )
    ref, ref_max, ref_lse = _reference(q.unsqueeze(1), [(kv, indices, lengths)], sink)
    torch.testing.assert_close(out, ref.squeeze(1), atol=0.02, rtol=0.02)
    torch.testing.assert_close(maximum, ref_max.squeeze(1), atol=0.003, rtol=0.003)
    torch.testing.assert_close(lse, ref_lse.squeeze(1), atol=0.003, rtol=0.003)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("extra_bits", [None, 8, 4])
def test_decode_mixed_lengths_and_strides(reverse, extra_bits):
    torch.manual_seed(12)
    q = torch.randn(3, 2, 64, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2]
    raw = torch.randn(259, 256, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed, decoded = _pack(raw, 8)
    indices = torch.randint(
        0, 259 * 256, (3, 2, 130), device="cuda", dtype=torch.int32
    )[..., ::2]
    indices[0, :, 0] = 65535
    indices[0, :, 1] = 65536
    indices[1, :, 0] = -1
    lengths = torch.tensor([65, 1, 0], dtype=torch.int32, device="cuda")
    if reverse:
        indices, lengths = indices.flip(0), lengths.flip(0)
    scopes = [(decoded, indices, lengths)]
    kwargs = {}
    if extra_bits is not None:
        extra, extra_decoded = _pack(raw[:2], extra_bits)
        ei = indices.remainder(512)
        el = torch.tensor([33, 0, 65], device="cuda", dtype=torch.int32)
        scopes.append((extra_decoded, ei, el))
        kwargs.update(
            extra_k_cache=extra, extra_indices_in_kvcache=ei, extra_topk_length=el
        )
    sink = torch.linspace(-2, 2, 64, device="cuda")
    sink[0], sink[1] = -torch.inf, torch.inf
    meta, _ = get_mla_metadata()
    out, lse = flash_mla_with_kvcache(
        q,
        packed,
        None,
        None,
        512,
        meta,
        indices=indices,
        attn_sink=sink,
        topk_length=lengths,
        **kwargs,
    )
    ref, _, ref_lse = _reference(q, scopes, sink)
    torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)
    torch.testing.assert_close(lse, ref_lse.transpose(1, 2), atol=0.003, rtol=0.003)


def test_graph_replay_reads_new_lengths():
    torch.manual_seed(13)
    q = torch.randn(2, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    packed, decoded = _pack(
        torch.randn(2, 256, 1, 512, device="cuda", dtype=torch.bfloat16), 8
    )
    indices = torch.randint(0, 512, (2, 1, 65), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([65, 1], device="cuda", dtype=torch.int32)
    meta, _ = get_mla_metadata()

    def run():
        return flash_mla_with_kvcache(
            q, packed, None, None, 512, meta, indices=indices, topk_length=lengths
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = run()
    for values in ([1, 65], [0, 33], [65, 0]):
        lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        graph.replay()
        ref, _, ref_lse = _reference(q, [(decoded, indices, lengths)], None)
        torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)
        torch.testing.assert_close(lse, ref_lse.transpose(1, 2), atol=0.003, rtol=0.003)


def test_cache_byte_offset_beyond_four_gib():
    if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        pytest.skip("the 64-bit byte-offset test requires 6 GiB of free GPU memory")
    page = 256
    token = 2**23 + 17
    assert token * 528 > 2**32
    cache = torch.empty(
        ((token + page) // page, page, 1, 528), dtype=torch.uint8, device="cuda"
    )
    rows = torch.stack(
        [torch.ones(512, device="cuda"), torch.full((512,), 3.0, device="cuda")]
    ).bfloat16()
    packed, decoded = _pack(rows.reshape(1, 2, 1, 512), 8)
    cache[0, 0].copy_(packed[0, 0])
    cache[token // page, token % page].copy_(packed[0, 1])
    q = torch.randn(2, 1, 64, 512, dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([[[token]], [[0]]], dtype=torch.int32, device="cuda")
    for reverse in (False, True):
        selected = indices.flip(0) if reverse else indices
        meta, _ = get_mla_metadata()
        out, _ = flash_mla_with_kvcache(
            q, cache, None, None, 512, meta, indices=selected
        )
        expected_rows = decoded[0, :, 0] if reverse else decoded[0, :, 0].flip(0)
        expected = expected_rows[:, None, None, :].expand_as(out)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_ue8m0_minimum_scale_is_not_zero():
    data = (
        torch.full((1, 1, 1, 512), 448.0, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    scales = torch.zeros((1, 1, 1, 16), dtype=torch.uint8, device="cuda")
    cache = torch.cat([data, scales], dim=-1)
    q = torch.zeros((1, 1, 64, 512), dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
    meta, _ = get_mla_metadata()
    out, _ = flash_mla_with_kvcache(q, cache, None, None, 512, meta, indices=indices)
    torch.testing.assert_close(out, torch.full_like(q, 448 * 2.0**-127), rtol=0, atol=0)
