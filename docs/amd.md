# Experimental ROCm backend

The AMD backend currently implements V4.1 sparse MQA prefill and decoding for
BF16 queries with head dimension 512. It is a correctness bring-up baseline,
not yet a complete DeepInfer backend or a performance-qualified service.

- Main decode cache: 528 bytes/token (512 E4M3 bytes + 16 UE8M0 scales).
- Optional extra cache: the same FP8 format, or 288 bytes/token (256 packed
  E2M1 bytes + 32 E4M3 scales). Even dimensions occupy the low nibble.
- Selected KV rows are dequantized inside the attention kernel, not expanded
  into an HBM-resident BF16 cache.
- Strided queries/indices and request-specific top-k lengths are supported.
- Empty attention rows return zero output, maximum=-inf and LSE=+inf.
- Graph replay reads updated lengths from device memory. This baseline does
  not retain a length-dependent split worklist.
- Sparse cache addressing uses int64 byte offsets.

Fused norm/RoPE/attention/cast and dense attention are not implemented on AMD.
Do not enable the CUDA fused path on this backend.

The initial development matrix is AMD GPU, ROCm 7.2, PyTorch 2.11 and
Triton 3.7. Neither AITER nor FlyDSL is imported or required.

Build and test from a ROCm PyTorch environment:

```bash
FLASH_MLA_BUILD_TARGET_PLATFORM=ROCM python setup.py bdist_wheel
python -m pytest tests/amd/test_sparse_mla.py -q
```

The large byte-offset regression needs 6 GiB free GPU memory. Model-level
precision, long-context quality, DSpark and service stability require separate
validation; kernel tests do not establish those properties.
