#define FLASH_MLA_AMD_REUSE_EXTRA 1
#include "fused_sparse_attention_valid_output.hip"

namespace flash_mla::amd::compact_attention {
// The packet preserves the exact per-query key order. Consumers on later
// layers only reuse packets when the compressed cache and index source agree.
#ifdef FLASH_MLA_AMD_DECODE_EXTRA_PACKET
constexpr int PacketMode = 5, PacketBytes = 64 * (1024 + 4);
#else
constexpr int PacketMode = 4, PacketBytes = 64 * (256 + 32 + 4);
#endif
__global__ __launch_bounds__(256) void pack_extra(Args p, uint8_t *out) {
    constexpr int BK = 64, Body = 256, Scale = 32, Packet = PacketBytes;
    const int row = blockIdx.y, tile = blockIdx.x, wave = threadIdx.x / 64, lane = threadIdx.x % 64;
    const int length = p.extra_lengths ? min(max(p.extra_lengths[row], 0), p.extra_topk) : p.extra_topk;
    const int col = tile * BK + lane;
    const int token = col < length ? p.extra_indices[int64_t(row) * p.extra_index_stride + col] : -1;
    const bool ok = token >= 0 && int64_t(token) < int64_t(p.extra_blocks) * p.extra_page_size;
    uint64_t address = 0;
    if (ok)
        address = reinterpret_cast<uint64_t>(p.extra + int64_t(token / p.extra_page_size) * p.extra_block_stride +
                                             int64_t(token % p.extra_page_size) * p.extra_token_stride);
    auto *dst = out + (int64_t(row) * gridDim.x + tile) * Packet;
#ifdef FLASH_MLA_AMD_DECODE_EXTRA_PACKET
#pragma unroll
    for (int key = wave; key < BK; key += 4) {
        const uint64_t src = uint64_t(__shfl(uint32_t(address), key, 64)) | (uint64_t(__shfl(uint32_t(address >> 32), key, 64)) << 32);
        const auto *bytes = reinterpret_cast<const uint8_t *>(src);
        const uint32_t data = src ? *reinterpret_cast<const uint32_t *>(bytes + lane * 4) : 0;
        const float scale = src ? e4(bytes[Body + lane / 2]) : 0.f;
        const U4 bits = {fp4_pair<0>(data, scale), fp4_pair<1>(data, scale), fp4_pair<2>(data, scale), fp4_pair<3>(data, scale)};
        *reinterpret_cast<U4 *>(dst + physical_row(key) * 1024 + lane * 16) = bits;
    }
#else
    auto copy = [&](uint8_t *target, int group, int key, int offset) {
        const uint64_t src = uint64_t(__shfl(uint32_t(address), key, 64)) | (uint64_t(__shfl(uint32_t(address >> 32), key, 64)) << 32);
        const U4 value = src ? *reinterpret_cast<const U4 *>(src + offset) : U4{};
        *reinterpret_cast<U4 *>(target + group * 1024 + lane * 16) = value;
    };
#pragma unroll
    for (int group = wave; group < BK * Body / 1024; group += 4)
        copy(dst, group, group * 4 + lane / 16, lane % 16 * 16);
    if (wave < 2)
        copy(dst + BK * Body, wave, wave * 32 + lane / 2, Body + (lane % 2) * 16);
#endif
    const bool full = __ballot(ok) == ~uint64_t(0);
    if (wave == 0)
        reinterpret_cast<int *>(dst + Packet - BK * 4)[lane] = full && lane == 0 ? 2 : int(ok);
}
} // namespace flash_mla::amd::compact_attention
extern "C" int fused_attention_packet_bytes() { return flash_mla::amd::compact_attention::PacketBytes; }
extern "C" int fused_attention_pack_extra(const flash_mla::amd::fused_sparse::Args *p, void *dst, void *stream) {
    using namespace flash_mla::amd::compact_attention;
    if (!p || !dst || p->rows <= 0 || p->extra_topk <= 0 || p->extra_topk > 512 || !p->extra || !p->extra_indices || p->extra_format != 4 ||
        p->extra_blocks < 1 || p->extra_page_size < 1 || p->extra_token_stride < 288 ||
        p->extra_block_stride < p->extra_page_size * p->extra_token_stride || p->extra_index_stride < p->extra_topk)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(pack_extra, dim3((p->extra_topk + 63) / 64, p->rows), dim3(256), 0, static_cast<hipStream_t>(stream), *p,
        static_cast<uint8_t *>(dst));
    return hipGetLastError();
}
extern "C" int fused_attention_reuse64_prepare(int h) {
    using namespace flash_mla::amd::compact_attention;
    if (h != 64)
        return hipErrorInvalidValue;
    int r = prepare<64, 64, false, false, false, PacketMode>();
    if (r)
        return r;
    r = prepare<64, 64, false, true, false, PacketMode>();
    if (r)
        return r;
    r = prepare<64, 64, true, false, false, PacketMode>();
    if (r)
        return r;
    return prepare<64, 64, true, true, false, PacketMode>();
}
extern "C" int fused_attention_reuse64_launch(const flash_mla::amd::fused_sparse::Args *p, const void *packet, int h, void *stream) {
    using namespace flash_mla::amd::compact_attention;
    if (!valid_args(p) || !packet || h != 64 || p->extra_format != 4 || p->extra_topk <= 0)
        return hipErrorInvalidValue;
    Args a = *p;
    a.extra = static_cast<const uint8_t *>(packet);
    a.extra_block_stride = int64_t((a.extra_topk + 63) / 64) * PacketBytes;
    const auto s = static_cast<hipStream_t>(stream);
    if (a.neox)
        return a.positions64 ? launch<64, 64, true, true, false, PacketMode>(a, s) : launch<64, 64, true, false, false, PacketMode>(a, s);
    return a.positions64 ? launch<64, 64, false, true, false, PacketMode>(a, s) : launch<64, 64, false, false, false, PacketMode>(a, s);
}

#ifdef FLASH_MLA_AMD_PRODUCE_EXTRA_PACKET
extern "C" int fused_attention_source64_prepare(int h) {
    using namespace flash_mla::amd::compact_attention;
    if (h != 64)
        return hipErrorInvalidValue;
    constexpr int F = PacketMode + 2;
    int r = prepare<64, 64, false, false, false, F>();
    if (r)
        return r;
    r = prepare<64, 64, false, true, false, F>();
    if (r)
        return r;
    r = prepare<64, 64, true, false, false, F>();
    if (r)
        return r;
    return prepare<64, 64, true, true, false, F>();
}
extern "C" int fused_attention_source64_launch(const flash_mla::amd::fused_sparse::Args *p, void *dst, int h, void *stream) {
    using namespace flash_mla::amd::compact_attention;
    if (!valid_args(p) || !dst || h != 64 || p->extra_format != 4 || p->extra_topk <= 0)
        return hipErrorInvalidValue;
    constexpr int F = PacketMode + 2;
    auto *out = static_cast<uint8_t *>(dst);
    auto s = static_cast<hipStream_t>(stream);
    if (p->neox)
        return p->positions64 ? launch<64, 64, true, true, false, F>(*p, s, out) : launch<64, 64, true, false, false, F>(*p, s, out);
    return p->positions64 ? launch<64, 64, false, true, false, F>(*p, s, out) : launch<64, 64, false, false, false, F>(*p, s, out);
}
#endif
