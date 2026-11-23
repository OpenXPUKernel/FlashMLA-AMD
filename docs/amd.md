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

Dense attention and the CUDA fused API are not implemented on AMD. Keep the
CUDA fused dispatch disabled on this backend.

The optional native HIP kernels in `csrc/hip_kernels` provide separate,
shape-guarded projection transforms and packed sparse attention. The selected
attention translation unit is `fused_sparse_attention_compact_unroll.hip`;
the conservative implementation remains available as its shared baseline.
Unused research translation units are not part of this backend.

Build these libraries with DeepInfer's `scripts/build_amd_artifacts.py`. That
command also writes an ABI/SHA256 manifest for its opt-in adapters. It does
not change this package's ordinary sparse-MLA API or enable CUDA kernels.

The initial development matrix is AMD GPU, ROCm 7.2, PyTorch 2.11 and
Triton 3.7. No external vendor runtime is imported or required.

Build and test from a ROCm PyTorch environment:

```bash
FLASH_MLA_BUILD_TARGET_PLATFORM=ROCM python setup.py bdist_wheel
python -m pytest tests/amd/test_sparse_mla.py -q
```

The large byte-offset regression needs 6 GiB free GPU memory. Model-level
precision, long-context quality, DSpark and service stability require separate
validation; kernel tests do not establish those properties.

### Shared compressed KV for sparse decode

The optional native entry point in `csrc/hip_kernels/fused_sparse_attention_reuse_extra.hip` packs the selected compressed keys in each query's original order. Call `fused_attention_packet_bytes` to size each 64-key packet and `fused_attention_pack_extra` to populate it. Use `fused_attention_reuse64_prepare` before graph capture and `fused_attention_reuse64_launch` on the caller's stream. Recreate packets when the selected keys or underlying KV change.

Compile this translation unit with `-O3 -std=c++20 -ffp-contract=off` and `-DFLASH_MLA_AMD_TWO_SOFTMAX`. The optional definitions `FLASH_MLA_AMD_ROW_ROTATE` and `FLASH_MLA_AMD_ROW_XOR4` select alternate lane permutations for the same ordered reductions.

`FLASH_MLA_AMD_REQUEST_ROW_GROUP=32` groups workgroup launches across nearby requests for target and draft decode. It changes the launch order while keeping each original query's inputs, key sequence and output row. Partial request groups use their actual size.
