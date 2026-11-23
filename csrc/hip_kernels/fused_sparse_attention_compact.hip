// Native AMD GPU compact attention. BF16/FP32 arithmetic and packed cache I/O.
// Half of Q stays in registers; padded LDS rows allow immediate-offset reads.
#include "fused_sparse_attention.hip"
#if defined(FLASH_MLA_AMD_CACHE_TABLE) || defined(FLASH_MLA_AMD_NATIVE_FP8_CACHE) || defined(FLASH_MLA_AMD_FP4_HALF_PAIR) ||               \
    defined(FLASH_MLA_AMD_FP4_FLOAT_PAIR)
#include "packed_cache_values.hip"
#endif
#ifndef FLASH_MLA_AMD_COMPACT_MARK
#define FLASH_MLA_AMD_COMPACT_MARK(index)                                                                                                  \
    do {                                                                                                                                   \
    } while (false)
#endif

namespace flash_mla::amd::compact_attention {
using namespace flash_mla::amd::fused_sparse;
using I2 = int32_t __attribute__((ext_vector_type(2)));
using S4 = int16_t __attribute__((ext_vector_type(4)));
#ifndef FLASH_MLA_AMD_COMPACT_KV_STRIDE
#define FLASH_MLA_AMD_COMPACT_KV_STRIDE 528
#endif
static constexpr int KSTRIDE = FLASH_MLA_AMD_COMPACT_KV_STRIDE;
static_assert(KSTRIDE >= 512 && KSTRIDE % 8 == 0);
template <int H, int BK> struct L {
#ifdef FLASH_MLA_AMD_COMPACT_WIDE32
    static constexpr int PARTS = 4, W = H / 16 * PARTS, T = W * 64, OUT = 512 / PARTS / 16;
#else
    static constexpr int PARTS = BK / 16, W = H / 16 * PARTS, T = W * 64, OUT = 512 / PARTS / 16;
#endif
#ifdef FLASH_MLA_AMD_REGISTER_QUERY
    static constexpr int Q = 0, KV = 0;
#else
    static constexpr int Q = 0, KV = H * 272 * 2;
#endif
#ifdef FLASH_MLA_AMD_OVERLAP_DECODE
    static constexpr int DecodedBuffers = 2;
#else
    static constexpr int DecodedBuffers = 1;
#endif
#ifdef FLASH_MLA_AMD_ORDERED_SCORE
    static constexpr int ScoreColumns = 24;
#else
    static constexpr int ScoreColumns = BK;
#endif
    static constexpr int SCORE = KV + DecodedBuffers * BK * KSTRIDE * 2, PROB = SCORE + H * ScoreColumns * 4;
#ifdef FLASH_MLA_AMD_TWO_SOFTMAX
    static constexpr int ProbabilityTiles = BK / 32;
#else
    static constexpr int ProbabilityTiles = 1;
#endif
    static constexpr int STATE = PROB + ProbabilityTiles * H * 32 * 2, VALID = STATE + H * (2 + ProbabilityTiles) * 4;
    static constexpr int RAW = (VALID + DecodedBuffers * BK * 4 + 255) / 256 * 256;
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
#ifdef FLASH_MLA_AMD_REGISTER_KV
    static constexpr int RAW_END = RAW;
#elif defined(FLASH_MLA_AMD_EARLY_KV_PREFETCH)
    static constexpr int RAW_END = RAW + 2 * BK * 544;
#else
    static constexpr int RAW_END = RAW + BK * 544;
#endif
#ifdef FLASH_MLA_AMD_STAGE_INDICES
    static constexpr int END = RAW_END + (640 + 512) * sizeof(int32_t);
#else
    static constexpr int END = RAW_END;
#endif
    static constexpr int BYTES = END < H * 512 * 2 ? H * 512 * 2 : END;
#else
    static constexpr int BYTES = RAW;
#endif
    static_assert(BYTES >= H * 512 * 2, "the final output reuses Q/KV storage");
};
__device__ __forceinline__ int qpos(int row, int col) {
#ifdef FLASH_MLA_AMD_QUERY_PLANES
    return row * 256 + (col ^ ((row & 7) * 16));
#else
    return row * 272 + col;
#endif
}
__device__ __forceinline__ int physical_row(int row) { return (row & ~31) | (row & 3) | (((row >> 3) & 3) << 2) | (((row >> 2) & 1) << 4); }
__device__ __forceinline__ int physical_kpos(int row, int col) {
#ifdef FLASH_MLA_AMD_COMPACT_KV_SWIZZLE
    col ^= ((row & 3) << 4) ^ (((row >> 3) & 1) << 6);
#endif
    return row * KSTRIDE + col;
}
__device__ __forceinline__ int kpos(int row, int col) { return physical_kpos(physical_row(row), col); }
template <int BK> __device__ __forceinline__ int spos(int row, int col) { return row * BK + (col ^ ((row & 7) << 2)); }
__device__ __forceinline__ I4 lds_vector(const __attribute__((address_space(3))) I4 *__restrict__ p) { return *p; }
__device__ __forceinline__ float lds_scalar(const __attribute__((address_space(3))) float *__restrict__ p) { return *p; }
__device__ __forceinline__ I2 lds_transpose(const __attribute__((address_space(3))) S4 *__restrict__ p) {
    return __builtin_bit_cast(I2, __builtin_amdgcn_ds_read_tr16_b64_v4i16(const_cast<__attribute__((address_space(3))) S4 *>(p)));
}
template <int CTRL> __device__ __forceinline__ float dpp(float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), CTRL, 0xf, 0xf, false));
}
template <int MASK> __device__ __forceinline__ float xor16(float x) {
#ifdef FLASH_MLA_AMD_ROW_ROTATE
    // Rotation by eight is XOR 8 within a 16-lane DPP row.
    if constexpr (MASK == 8)
        return dpp<0x128>(x);
#else
    if constexpr (MASK == 8)
        return dpp<0x141>(dpp<0x140>(x));
#endif
#ifdef FLASH_MLA_AMD_ROW_XOR4
    if constexpr (MASK == 4)
        return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), (4 << 10) | 31));
#else
    if constexpr (MASK == 4)
        return dpp<0x1b>(dpp<0x141>(x));
#endif
    if constexpr (MASK == 2)
        return dpp<0x4e>(x);
    if constexpr (MASK == 1)
        return dpp<0xb1>(x);
}
__device__ __forceinline__ void sync_current_lds() {
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
    // Only the disjoint raw-row prefetch may remain in flight. All reads and
    // writes of the current decoded tile must complete before its barrier.
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
#ifdef FLASH_MLA_AMD_WAVE_ROW_DMA
#ifdef FLASH_MLA_AMD_NARROW_TILE_ADDRESS
// Valid rows entirely inside one 4-GiB segment share a scalar buffer resource.
// Only their low offset words cross lanes for the packed-body and scale planes.
template <int BK, int W, bool FP4>
__device__ __forceinline__ void prefetch_buffer_planes(uint8_t *raw, uint64_t base, uint32_t offset, int wave, int lane) {
    constexpr int Body = FP4 ? 256 : 512, Scale = FP4 ? 32 : 16;
    using LDS = __attribute__((address_space(3))) void *;
    const auto res = __builtin_amdgcn_make_buffer_rsrc(reinterpret_cast<uint8_t *>(base), short(0), 0xffffffffu, 0x00020000);
#pragma unroll
    for (int group = wave; group < BK * Body / 1024; group += W) {
        const int kr = group * (1024 / Body) + lane / (Body / 16);
        const uint32_t off = __shfl(offset, kr, 64) + lane % (Body / 16) * 16;
        __builtin_amdgcn_raw_ptr_buffer_load_lds(res, (LDS)(raw + group * 1024), 16, off, 0, 0, 0);
    }
    if (wave < (BK * Scale + 1023) / 1024) {
        const int kr = wave * (1024 / Scale) + lane / (Scale / 16);
        if (kr < BK) {
            const uint32_t off = __shfl(offset, kr, 64) + Body + lane % (Scale / 16) * 16;
            __builtin_amdgcn_raw_ptr_buffer_load_lds(res, (LDS)(raw + BK * Body + wave * 1024), 16, off, 0, 0, 0);
        }
    }
    if (wave == 0 && lane < BK)
        reinterpret_cast<int *>(raw + BK * (Body + Scale))[lane] = lane == 0 ? 2 : 1;
}
#endif
template <int BK, int W, bool FP4, bool Full = false>
__device__ __forceinline__ void prefetch_row_planes(uint8_t *raw, uint64_t row_address, int wave, int lane) {
    constexpr int Body = FP4 ? 256 : 512, Scale = FP4 ? 32 : 16;
    using Global = const __attribute__((address_space(1))) uint8_t *;
    using Local = __attribute__((address_space(3))) void *;
    // Each issuing wave fills 1024 contiguous LDS bytes. Its lanes select
    // independent source rows with full 64-bit addresses.
    auto transfer = [&](uint8_t *dest, int group, int kr, int offset) {
        const uint64_t chosen =
            uint64_t(__shfl(uint32_t(row_address), kr, 64)) | (uint64_t(__shfl(uint32_t(row_address >> 32), kr, 64)) << 32);
        if (Full || chosen)
            __builtin_amdgcn_global_load_lds(reinterpret_cast<Global>(chosen + offset), (Local)(dest + group * 1024), 16, 0, 0);
        else
            *reinterpret_cast<U4 *>(dest + group * 1024 + lane * 16) = U4{};
    };
#pragma unroll
    for (int group = wave; group < BK * Body / 1024; group += W) {
        const int kr = group * (1024 / Body) + lane / (Body / 16);
        transfer(raw, group, kr, lane % (Body / 16) * 16);
    }
    constexpr int ScaleWaves = (BK * Scale + 1023) / 1024;
    if (wave < ScaleWaves) {
        const int kr = wave * (1024 / Scale) + lane / (Scale / 16);
        if (kr < BK)
            transfer(raw + BK * Body, wave, kr, Body + lane % (Scale / 16) * 16);
    }
    if (wave == 0 && lane < BK)
        reinterpret_cast<int *>(raw + BK * (Body + Scale))[lane] = Full ? (lane == 0 ? 2 : 1) : int(row_address != 0);
}
#endif
#ifdef FLASH_MLA_AMD_REGISTER_KV
struct PackedRowFragment {
    uint32_t lo, hi, sf;
    int valid;
};
template <int BK, int W, bool CONTIG>
__device__ __forceinline__ void load_row_fragments(
    const Args &p, PackedRowFragment (&v)[BK / W], int row, int tile, int mt, int ml, int el, int wave, int lane) {
    const bool extra = tile >= mt, fp4 = extra && p.extra_format == 4;
    const int body = fp4 ? 256 : 512, bytes = fp4 ? 288 : 528, length = extra ? el : ml, begin = (extra ? tile - mt : tile) * BK;
    const auto *indices = (extra ? p.extra_indices : p.indices) + int64_t(row) * (extra ? p.extra_index_stride : p.index_stride);
    const auto *cache = extra ? p.extra : p.cache;
    const int blocks = extra ? p.extra_blocks : p.cache_blocks, page = extra ? p.extra_page_size : p.page_size;
    const int64_t bs = extra ? p.extra_block_stride : p.cache_block_stride, ts = extra ? p.extra_token_stride : p.cache_token_stride;
    const int fetched = lane < BK && begin + lane < length ? indices[begin + lane] : -1;
#pragma unroll
    for (int b = 0; b < BK / W; ++b) {
        const int token = __builtin_amdgcn_readlane(fetched, wave + b * W);
        const bool ok = token >= 0 && int64_t(token) < int64_t(blocks) * page;
        const int at = ok ? token : 0;
        const auto *src = CONTIG ? cache + int64_t(at) * bytes : cache + int64_t(at / page) * bs + int64_t(at % page) * ts;
        const auto res = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0), ok ? bytes : 0, 0x00020000);
        v[b].valid = ok;
        v[b].lo = __builtin_amdgcn_raw_buffer_load_b32(res, lane * (fp4 ? 4 : 8), 0, 0);
        if (!fp4)
            v[b].hi = __builtin_amdgcn_raw_buffer_load_b32(res, lane * 8 + 4, 0, 0);
        v[b].sf = __builtin_amdgcn_raw_buffer_load_b32(res, body + ((lane / (fp4 ? 2 : 4)) & ~3), 0, 0);
    }
}
#endif
template <int H, int BK, int W = L<H, BK>::W, bool CONTIG = false, int FIXED = 0>
__device__ __forceinline__ void prefetch_raw(
    const Args &p, uint8_t *raw, int row, int tile, int mt, int ml, int el, int wave, int lane, int prepared = -1) {
    const bool extra = tile >= mt;
#ifdef FLASH_MLA_AMD_REUSE_EXTRA
    if constexpr (FIXED == 5)
        if (extra)
            return;
    if constexpr (FIXED == 4) {
        if (extra) {
            constexpr int Packet = BK * (256 + 32 + 4);
            const auto *src = p.extra + int64_t(row) * p.extra_block_stride + int64_t(tile - mt) * Packet;
            auto res = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0), Packet, 0x00020000);
            using Local = __attribute__((address_space(3))) void *;
#pragma unroll
            for (int group = wave; group < (Packet + 1023) / 1024; group += W)
                if (group * 1024 + lane * 16 < Packet)
                    __builtin_amdgcn_raw_ptr_buffer_load_lds(res, (Local)(raw + group * 1024), 16, lane * 16, group * 1024, 0, 0);
            return;
        }
    }
#endif
    const int begin = (extra ? tile - mt : tile) * BK, length = extra ? el : ml;
    const int32_t *index = (extra ? p.extra_indices : p.indices) + int64_t(row) * (extra ? p.extra_index_stride : p.index_stride);
    const uint8_t *cache = extra ? p.extra : p.cache;
    const int page = extra ? p.extra_page_size : p.page_size, blocks = extra ? p.extra_blocks : p.cache_blocks;
    const int64_t bs = extra ? p.extra_block_stride : p.cache_block_stride, ts = extra ? p.extra_token_stride : p.cache_token_stride;
    const int bytes = extra && p.extra_format == 4 ? 288 : 528;
#ifdef FLASH_MLA_AMD_COMPACT_LINEAR_ADDRESS
    const bool contiguous = ts == bytes && bs == int64_t(page) * bytes;
#endif
    // Fetch this tile's indices before issuing any asynchronous row transfer.
    // Otherwise the next index load can force completion of the previous DMA.
#if defined(FLASH_MLA_AMD_STAGE_INDICES) || defined(FLASH_MLA_AMD_LOOKAHEAD_INDEX)
    int fetched = prepared;
#else
    int fetched = lane < BK && begin + lane < length ? index[begin + lane] : -1;
#endif
#ifdef FLASH_MLA_AMD_WAVE_ROW_DMA
    const bool valid_token = fetched >= 0 && int64_t(fetched) < int64_t(blocks) * page;
    uint64_t address = 0;
    if (valid_token) {
        if constexpr (FIXED >= 1 && FIXED <= 3) {
            const uint32_t token = uint32_t(fetched);
            const uint32_t physical = extra ? token + (token >> 6) * (396 - 64) : token + (token / 133u) * (5860 - 133);
            address = reinterpret_cast<uint64_t>(cache + uint64_t(physical) * (extra ? 288 : 528));
        } else if constexpr (CONTIG)
            address = reinterpret_cast<uint64_t>(cache + int64_t(fetched) * bytes);
        else
            address = reinterpret_cast<uint64_t>(cache + int64_t(fetched / page) * bs + int64_t(fetched % page) * ts);
    }
#ifdef FLASH_MLA_AMD_FULL_VALID_TILE
    constexpr uint64_t required = ~uint64_t(0) >> (64 - BK);
    const bool full = __ballot(valid_token) == required;
#ifdef FLASH_MLA_AMD_NARROW_TILE_ADDRESS
    if (full) {
        const uint64_t relative = address - reinterpret_cast<uint64_t>(cache);
        const uint32_t lower = uint32_t(relative), upper = uint32_t(relative >> 32);
        const uint32_t common = uint32_t(__builtin_amdgcn_readfirstlane(upper));
        if (!__ballot(lane < BK && (upper != common || lower > 0xffffffffu - uint32_t(bytes)))) {
            const uint64_t base = reinterpret_cast<uint64_t>(cache) + (uint64_t(common) << 32);
            if (extra && p.extra_format == 4)
                prefetch_buffer_planes<BK, W, true>(raw, base, lower, wave, lane);
            else
                prefetch_buffer_planes<BK, W, false>(raw, base, lower, wave, lane);
            return;
        }
    }
#endif
    if (full) {
        if (extra && p.extra_format == 4)
            prefetch_row_planes<BK, W, true, true>(raw, address, wave, lane);
        else
            prefetch_row_planes<BK, W, false, true>(raw, address, wave, lane);
    } else
#endif
    {
        if (extra && p.extra_format == 4)
            prefetch_row_planes<BK, W, true>(raw, address, wave, lane);
        else
            prefetch_row_planes<BK, W, false>(raw, address, wave, lane);
    }
    return;
#endif
#ifdef FLASH_MLA_AMD_COMPACT_DIRECT_LANE
    asm volatile("v_mov_b32 %0, %0" : "+v"(fetched));
#endif
#ifdef FLASH_MLA_AMD_COMPACT_VECTOR_ADDRESS
    // All selected rows are independent. Prepare their addresses in parallel,
    // then broadcast only each row descriptor to its DMA wave.
    const int fetched_ok = fetched >= 0 && int64_t(fetched) < int64_t(blocks) * page;
    const int chosen = fetched_ok ? fetched : 0;
    const int block = page == 128 ? chosen >> 7 : chosen / page, offset = page == 128 ? chosen & 127 : chosen % page;
    const uint64_t address = reinterpret_cast<uint64_t>(cache + int64_t(block) * bs + int64_t(offset) * ts);
#endif
#ifdef FLASH_MLA_AMD_COMPACT_UNROLL_ROWS
#pragma unroll
#else
#pragma unroll 2
#endif
    for (int b = 0; b < BK / W; ++b) {
        const int kr = wave + b * W;
#ifdef FLASH_MLA_AMD_COMPACT_VECTOR_ADDRESS
        const bool ok = __builtin_amdgcn_readfirstlane(__shfl(fetched_ok, kr, 64));
        const uint32_t lo = __builtin_amdgcn_readfirstlane(__shfl(uint32_t(address), kr, 64));
        const uint32_t hi = __builtin_amdgcn_readfirstlane(__shfl(uint32_t(address >> 32), kr, 64));
        const uint8_t *src = reinterpret_cast<const uint8_t *>(uint64_t(lo) | (uint64_t(hi) << 32));
#else
#ifdef FLASH_MLA_AMD_COMPACT_DIRECT_LANE
        const int token = __builtin_amdgcn_readlane(fetched, kr);
#else
        const int token = __builtin_amdgcn_readfirstlane(__shfl(fetched, kr, 64));
#endif
        const bool ok = token >= 0 && int64_t(token) < int64_t(blocks) * page;
        const int chosen = ok ? token : 0;
        const uint8_t *src;
        if constexpr (CONTIG)
            src = cache + int64_t(chosen) * bytes;
        else {
#ifdef FLASH_MLA_AMD_COMPACT_LINEAR_ADDRESS
            if (contiguous)
                src = cache + int64_t(chosen) * bytes;
            else {
#endif
                const int block = page == 128 ? chosen >> 7 : chosen / page, offset = page == 128 ? chosen & 127 : chosen % page;
                src = cache + int64_t(block) * bs + int64_t(offset) * ts;
#ifdef FLASH_MLA_AMD_COMPACT_LINEAR_ADDRESS
            }
#endif
        }
#endif
        const auto resource = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0), ok ? bytes : 0, 0x00020000);
        using LDS = __attribute__((address_space(3))) void *;
        if (lane < bytes / 16)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(resource, (LDS)(raw + kr * 544), 16, lane * 16, 0, 0, 0);
        if (lane == 0)
            *reinterpret_cast<int *>(raw + kr * 544 + 540) = int(ok);
    }
}
#endif
__device__ __forceinline__ I4 read_v(const uint16_t *kv, int col, int lane, int part) {
    using LDS = __attribute__((address_space(3))) S4 *;
    const int row = part * 32 + (lane >> 2), c = col + (lane & 3) * 4;
    const auto a = lds_transpose((LDS)(kv + physical_kpos(row, c)));
    const auto b = lds_transpose((LDS)(kv + physical_kpos(row + 16, c)));
    return I4{a[0], a[1], b[0], b[1]};
}
#ifdef FLASH_MLA_AMD_OVERLAP_DECODE
template <int BK, int Readers>
__device__ __forceinline__ void decode_plane(const uint8_t *raw, uint16_t *kv, int *valid, bool fp4, int reader, int lane) {
    const int body = fp4 ? 256 : 512, scale_bytes = fp4 ? 32 : 16, col = lane * 8;
#pragma unroll
    for (int kr = reader; kr < BK; kr += Readers) {
        const auto *p = raw + kr * body;
        const auto *sf = raw + BK * body + kr * scale_bytes;
        if (lane == 0)
            valid[kr] = reinterpret_cast<const int *>(raw + BK * (body + scale_bytes))[kr];
        U4 bits;
        if (fp4) {
            const auto data = *reinterpret_cast<const uint32_t *>(p + col / 2);
            const float scale = e4(sf[col / 16]);
            bits = U4{fp4_pair<0>(data, scale), fp4_pair<1>(data, scale), fp4_pair<2>(data, scale), fp4_pair<3>(data, scale)};
        } else {
            const auto data = *reinterpret_cast<const uint64_t *>(p + col);
            const float scale = ue8(sf[col / 32]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const auto val = __builtin_amdgcn_cvt_pk_f32_fp8(uint32_t(data >> (j * 16)), false);
                bits[j] = pack2(val[0] * scale, val[1] * scale);
            }
        }
        *reinterpret_cast<U4 *>(kv + kpos(kr, col)) = bits;
    }
}
#endif
template <bool NEOX> __device__ __forceinline__ void rotate_tail(I4 &lo, I4 &hi, const float *cs, int g) {
    const U4 ra = __builtin_bit_cast(U4, lo), rb = __builtin_bit_cast(U4, hi);
    float a[8], b[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[j * 2] = bf(ra[j]);
        a[j * 2 + 1] = bf(ra[j] >> 16);
        b[j * 2] = bf(rb[j]);
        b[j * 2 + 1] = bf(rb[j] >> 16);
    }
    if constexpr (NEOX) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float c = cs[g * 8 + j], s = cs[32 + g * 8 + j], x = a[j], y = b[j];
            a[j] = fmaf(x, c, -y * s);
            b[j] = fmaf(x, s, y * c);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            float c = cs[g * 4 + j / 2], s = cs[32 + g * 4 + j / 2], x = a[j], y = a[j + 1];
            a[j] = fmaf(x, c, -y * s);
            a[j + 1] = fmaf(x, s, y * c);
            c = cs[16 + g * 4 + j / 2];
            s = cs[48 + g * 4 + j / 2];
            x = b[j];
            y = b[j + 1];
            b[j] = fmaf(x, c, -y * s);
            b[j + 1] = fmaf(x, s, y * c);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo[j] = pack2(a[j * 2], a[j * 2 + 1]);
        hi[j] = pack2(b[j * 2], b[j * 2 + 1]);
    }
}

template <int H, int BK, bool NEOX, bool POS64, bool CONTIG = false, int FIXED = 0>
__global__ __launch_bounds__((L<H, BK>::T)) void kernel(Args p, uint8_t *packet_out) {
    if constexpr (FIXED >= 1 && FIXED <= 3) {
        p.page_size = 133;
        p.cache_token_stride = 528;
        p.cache_block_stride = 3094080;
        p.extra_page_size = 64;
        p.extra_token_stride = 288;
        p.extra_block_stride = 114048;
        p.topk = FIXED == 3 ? 132 : 128;
        p.extra_topk = FIXED == 2 ? 512 : 0;
        p.extra_format = 4;
    }
    constexpr int W = L<H, BK>::W, T = L<H, BK>::T, PARTS = L<H, BK>::PARTS;
    extern __shared__ __attribute__((aligned(16))) uint8_t shared[];
    auto *q = reinterpret_cast<uint16_t *>(shared);
    auto *kv = reinterpret_cast<uint16_t *>(shared + L<H, BK>::KV);
    auto *score = reinterpret_cast<float *>(shared + L<H, BK>::SCORE);
#ifdef FLASH_MLA_AMD_COMPACT_RELOAD_SCORE
    auto *reductions = score + (BK == 64 ? H * 32 : 0);
#else
    auto *reductions = score;
#endif
    auto *prob = reinterpret_cast<uint16_t *>(shared + L<H, BK>::PROB);
    auto *state = reinterpret_cast<float *>(shared + L<H, BK>::STATE);
    auto *valid = reinterpret_cast<int *>(shared + L<H, BK>::VALID);
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
    auto *raw = shared + L<H, BK>::RAW;
#endif
    const int tid = threadIdx.x, lane = tid % 64, r = lane % 16, g = lane / 16, wave = __builtin_amdgcn_readfirstlane(tid / 64);
    constexpr int QKW = H / 16 * (BK / 16);
    const int qw = wave % QKW;
    const int row = blockIdx.x, hb = blockIdx.y * H, qh = qw / (BK / 16) * 16 + r, kh = qw % (BK / 16) * 16 + r;
    const int ml = p.lengths ? min(max(p.lengths[row], 0), p.topk) : p.topk;
    const int el = p.extra_lengths ? min(max(p.extra_lengths[row], 0), p.extra_topk) : p.extra_topk;
    const int mt = (ml + BK - 1) / BK, et = (el + BK - 1) / BK;
#ifdef FLASH_MLA_AMD_STAGE_INDICES
    auto *indices_lds = reinterpret_cast<int32_t *>(shared + L<H, BK>::RAW_END);
    for (int col = tid; col < ml; col += T)
        indices_lds[col] = p.indices[int64_t(row) * p.index_stride + col];
    for (int col = tid; col < el; col += T)
        indices_lds[640 + col] = p.extra_indices[int64_t(row) * p.extra_index_stride + col];
#endif
    FLASH_MLA_AMD_COMPACT_MARK(0);
    const int64_t pos = POS64 ? static_cast<const int64_t *>(p.positions)[row] : static_cast<const int32_t *>(p.positions)[row];
    const float *cs = p.cos_sin + pos * p.cos_sin_stride;
    if (p.invocations && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        atomicAdd(p.invocations, 1ULL);
#ifdef FLASH_MLA_AMD_COOPERATIVE_QUERY
#ifdef FLASH_MLA_AMD_QUERY_PLANES
    // Two consecutive heads fill one 1024-byte lane transfer per half.
    // The high halves occupy otherwise unused KV workspace until Q is ready.
    using QGlobal = const __attribute__((address_space(1))) uint8_t *;
    using QLocal = __attribute__((address_space(3))) void *;
#pragma unroll
    for (int pair = wave; pair < H / 2; pair += W) {
        const int h = pair * 2 + lane / 32, col = (lane % 32) * 8 ^ ((h & 7) * 16);
        const uint16_t *src = p.q + int64_t(row) * p.q_stride + (hb + h) * 512 + col;
        __builtin_amdgcn_global_load_lds(reinterpret_cast<QGlobal>(reinterpret_cast<uintptr_t>(src)), (QLocal)(q + pair * 512), 16, 0, 0);
        __builtin_amdgcn_global_load_lds(
            reinterpret_cast<QGlobal>(reinterpret_cast<uintptr_t>(src + 256)), (QLocal)(kv + pair * 512), 16, 0, 0);
    }
    __builtin_amdgcn_s_waitcnt(0xf70);
#else
    for (int h = wave; h < H; h += W) {
        const U4 value = *reinterpret_cast<const U4 *>(p.q + int64_t(row) * p.q_stride + (hb + h) * 512 + lane * 8);
        if (lane < 32)
            *reinterpret_cast<U4 *>(q + qpos(h, lane * 8)) = value;
        else
            *reinterpret_cast<U4 *>(kv + h * 256 + (lane - 32) * 8) = value;
    }
#endif
    sync_current_lds();
#else
    for (int h = wave; h < H; h += W) {
        if (lane < 32)
            *reinterpret_cast<U4 *>(q + qpos(h, lane * 8)) =
                *reinterpret_cast<const U4 *>(p.q + int64_t(row) * p.q_stride + (hb + h) * 512 + lane * 8);
    }
#endif
    I4 high[8];
    if (wave < QKW) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#ifdef FLASH_MLA_AMD_COOPERATIVE_QUERY
#ifdef FLASH_MLA_AMD_QUERY_PLANES
            const int offset = qpos(qh, k * 32 + g * 8);
#else
            const int offset = qh * 256 + k * 32 + g * 8;
#endif
            high[k] = lds_vector((const __attribute__((address_space(3))) I4 *)(kv + offset));
            asm volatile("" : "+v"(high[k][0]), "+v"(high[k][1]), "+v"(high[k][2]), "+v"(high[k][3])::"memory");
#else
            high[k] = *reinterpret_cast<const I4 *>(p.q + int64_t(row) * p.q_stride + (hb + qh) * 512 + 256 + k * 32 + g * 8);
#endif
        }
        if (!p.already_roped)
            rotate_tail<NEOX>(high[6], high[7], cs, g);
#if defined(FLASH_MLA_AMD_COMPACT_PIPELINE) && !defined(FLASH_MLA_AMD_COOPERATIVE_QUERY)
        // Resolve the original global Q loads before the prefetch loop. Explicit
        // register definitions avoid loop-carried conservative VMEM dependencies.
#pragma unroll
        for (int k = 0; k < 8; ++k)
            asm volatile("v_mov_b32 %0, %0\n\tv_mov_b32 %1, %1\n\tv_mov_b32 %2, %2\n\tv_mov_b32 %3, %3"
                : "+v"(high[k][0]), "+v"(high[k][1]), "+v"(high[k][2]), "+v"(high[k][3]));
#endif
    }
    if (tid < H) {
        state[tid * 3] = -INFINITY;
        state[tid * 3 + 1] = 0;
        state[tid * 3 + 2] = 0;
    }
    sync_current_lds();
#ifdef FLASH_MLA_AMD_REGISTER_QUERY
    I4 low[8];
    if (wave < QKW) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            low[k] = lds_vector((const __attribute__((address_space(3))) I4 *)(q + qpos(qh, k * 32 + g * 8)));
            asm volatile("" : "+v"(low[k][0]), "+v"(low[k][1]), "+v"(low[k][2]), "+v"(low[k][3])::"memory");
        }
    }
    // All waves have taken their query fragments before KV reuses this LDS.
    sync_current_lds();
#endif
    FLASH_MLA_AMD_COMPACT_MARK(1);
    F4 output[L<H, BK>::OUT] = {};
#ifdef FLASH_MLA_AMD_RUNNING_ROW_STATE
    float running_max = -INFINITY, running_sum = 0.f;
#endif
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
#ifdef FLASH_MLA_AMD_PREDECODED_SWA
    const bool decoded_swa = p.cache_token_stride == -1024 && p.cache_block_stride == int64_t(p.page_size) * 1024;
#else
    constexpr bool decoded_swa = false;
#endif
#if defined(FLASH_MLA_AMD_STAGE_INDICES) || defined(FLASH_MLA_AMD_LOOKAHEAD_INDEX)
    auto index_for = [&](int tile) {
        if (tile >= mt + et || lane >= BK)
            return -1;
        const bool extra = tile >= mt;
#ifdef FLASH_MLA_AMD_REUSE_EXTRA
        if constexpr (FIXED == 4 || FIXED == 5)
            if (extra)
                return -1;
#endif
        const int col = (extra ? tile - mt : tile) * BK + lane;
        if (col >= (extra ? el : ml))
            return -1;
#ifdef FLASH_MLA_AMD_STAGE_INDICES
        return indices_lds[(extra ? 640 : 0) + col];
#else
        return extra ? p.extra_indices[int64_t(row) * p.extra_index_stride + col] : p.indices[int64_t(row) * p.index_stride + col];
#endif
    };
#endif
#ifdef FLASH_MLA_AMD_LOOKAHEAD_INDEX
    int ahead = index_for(0);
#endif
#ifdef FLASH_MLA_AMD_REGISTER_KV
    PackedRowFragment pending[BK / W];
#endif
    auto prefetch = [&](uint8_t *target, int tile) {
#ifdef FLASH_MLA_AMD_LOOKAHEAD_INDEX
        const int index = ahead;
        ahead = index_for(tile + 1);
#elif defined(FLASH_MLA_AMD_STAGE_INDICES)
        const int index = index_for(tile);
#else
        constexpr int index = -1;
#endif
#ifdef FLASH_MLA_AMD_REGISTER_KV
        load_row_fragments<BK, W, CONTIG>(p, pending, row, tile, mt, ml, el, wave, lane);
#else
        prefetch_raw<H, BK, L<H, BK>::W, CONTIG, FIXED>(p, target, row, tile, mt, ml, el, wave, lane, index);
#endif
    };
    if (mt + et && (!decoded_swa || mt == 0)) {
        prefetch(raw, 0);
        __builtin_amdgcn_s_waitcnt(0xf70);
        sync_current_lds();
    }
#endif
#ifdef FLASH_MLA_AMD_OVERLAP_DECODE
    static_assert(BK == 32 && W == QKW * 2);
    if (mt + et) {
        decode_plane<BK, W>(raw, kv, valid, mt == 0 && p.extra_format == 4, wave, lane);
        sync_current_lds();
        if (mt + et > 1) {
            prefetch(raw, 1);
            __builtin_amdgcn_s_waitcnt(0xf70);
            sync_current_lds();
        }
    }
#endif
    FLASH_MLA_AMD_COMPACT_MARK(2);
#ifdef FLASH_MLA_AMD_SEPARATE_CACHE_LOOPS
    auto consume = [&](int tile, auto extra_tag, auto complete_tag) {
        constexpr bool extra = decltype(extra_tag)::value, whole = decltype(complete_tag)::value;
#else
    for (int tile = 0; tile < mt + et; ++tile) {
        const bool extra = tile >= mt;
        constexpr bool whole = false;
#endif
#ifdef FLASH_MLA_AMD_OVERLAP_DECODE
        auto *kv_base = reinterpret_cast<uint16_t *>(shared + L<H, BK>::KV);
        auto *valid_base = reinterpret_cast<int *>(shared + L<H, BK>::VALID);
        auto *kv = kv_base + (tile & 1) * BK * KSTRIDE;
        auto *valid = valid_base + (tile & 1) * BK;
        if (wave >= QKW && tile + 1 < mt + et) {
            decode_plane<BK, W - QKW>(raw, kv_base + ((tile + 1) & 1) * BK * KSTRIDE, valid_base + ((tile + 1) & 1) * BK,
                tile + 1 >= mt && p.extra_format == 4, wave - QKW, lane);
        }
#endif
#ifdef FLASH_MLA_AMD_EARLY_KV_PREFETCH
        uint8_t *const raw = shared + L<H, BK>::RAW + (tile & 1) * BK * 544;
        if (tile + 1 < mt + et) {
            uint8_t *const next = shared + L<H, BK>::RAW + ((tile + 1) & 1) * BK * 544;
            prefetch(next, tile + 1);
        }
#endif
        const int begin = (extra ? tile - mt : tile) * BK, length = extra ? el : ml;
        const int32_t *index = (extra ? p.extra_indices : p.indices) + int64_t(row) * (extra ? p.extra_index_stride : p.index_stride);
        const uint8_t *cache = extra ? p.extra : p.cache;
        const int page = extra ? p.extra_page_size : p.page_size, blocks = extra ? p.extra_blocks : p.cache_blocks;
        const int64_t bs = extra ? p.extra_block_stride : p.cache_block_stride, ts = extra ? p.extra_token_stride : p.cache_token_stride;
        const bool fp4 = extra && p.extra_format == 4;
#ifndef FLASH_MLA_AMD_OVERLAP_DECODE
#ifdef FLASH_MLA_AMD_REGISTER_KV
#pragma unroll
        for (int b = 0; b < BK / W; ++b) {
            const int kr = wave + b * W, col = lane * 8;
            const auto row_fragment = pending[b];
            if (lane == 0)
                valid[kr] = row_fragment.valid;
            U4 bits;
            if (fp4) {
                const float scale = e4(uint8_t(row_fragment.sf >> ((lane / 2 % 4) * 8)));
                bits = U4{fp4_pair<0>(row_fragment.lo, scale), fp4_pair<1>(row_fragment.lo, scale), fp4_pair<2>(row_fragment.lo, scale),
                    fp4_pair<3>(row_fragment.lo, scale)};
            } else {
                const float scale = ue8(uint8_t(row_fragment.sf >> ((lane / 4 % 4) * 8)));
                const uint64_t data = uint64_t(row_fragment.lo) | (uint64_t(row_fragment.hi) << 32);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const auto v = __builtin_amdgcn_cvt_pk_f32_fp8(uint32_t(data >> (j * 16)), false);
                    bits[j] = pack2(v[0] * scale, v[1] * scale);
                }
            }
            *reinterpret_cast<U4 *>(kv + kpos(kr, col)) = bits;
        }
#else
#ifdef FLASH_MLA_AMD_DECODE_EXTRA_PACKET
        if (FIXED == 5 && extra) {
            constexpr int Packet = BK * (1024 + 4);
            const auto *src = p.extra + int64_t(row) * p.extra_block_stride + int64_t(tile - mt) * Packet;
            auto res = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0), Packet, 0x00020000);
            using Local = __attribute__((address_space(3))) void *;
#pragma unroll
            for (int physical = wave; physical < BK; physical += W)
                __builtin_amdgcn_raw_ptr_buffer_load_lds(res, (Local)(kv + physical * KSTRIDE), 16, lane * 16, physical * 1024, 0, 0);
            if (wave == 0 && lane < BK / 4)
                __builtin_amdgcn_raw_ptr_buffer_load_lds(res, (Local)valid, 16, lane * 16, BK * 1024, 0, 0);
            __builtin_amdgcn_s_waitcnt(0xf70);
        } else
#endif
#ifdef FLASH_MLA_AMD_PREDECODED_SWA
            if (decoded_swa && !extra) {
            // Each wave copies one decoded key row directly into its transposed
            // row slot. The same packed -> BF16 rounding was performed in prepare.
#pragma unroll
            for (int b = 0; b < BK / W; ++b) {
                const int kr = wave + b * W;
                const int token = __builtin_amdgcn_readfirstlane(begin + kr < length ? index[begin + kr] : -1);
                const bool ok = token >= 0 && int64_t(token) < int64_t(blocks) * page;
                const uint8_t *src = cache + int64_t(ok ? token : 0) * 1024;
                auto res = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0), ok ? 1024 : 0, 0x00020000);
                using LDS = __attribute__((address_space(3))) void *;
                __builtin_amdgcn_raw_ptr_buffer_load_lds(res, (LDS)(kv + kpos(kr, 0)), 16, lane * 16, 0, 0, 0);
                if (lane == 0)
                    valid[kr] = int(ok);
            }
            __builtin_amdgcn_s_waitcnt(0xf70);
        } else
#endif
#ifdef FLASH_MLA_AMD_COMPACT_UNROLL_ROWS
#pragma unroll
#else
#pragma unroll 2
#endif
            for (int b = 0; b < BK / W; ++b) {
                const int kr = wave + b * W, col = lane * 8;
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
#ifdef FLASH_MLA_AMD_WAVE_ROW_DMA
                const int body = fp4 ? 256 : 512, scale_bytes = fp4 ? 32 : 16;
                const uint8_t *packed = raw + kr * body;
                const uint8_t *packed_scale = raw + BK * body + kr * scale_bytes;
                if (lane == 0)
                    valid[kr] = reinterpret_cast<const int *>(raw + BK * (body + scale_bytes))[kr];
#else
                const uint8_t *packed = raw + kr * 544;
                const uint8_t *packed_scale = packed + (fp4 ? 256 : 512);
                if (lane == 0)
                    valid[kr] = *reinterpret_cast<const int *>(packed + 540);
#endif
#else
            const int token = __builtin_amdgcn_readfirstlane(begin + kr < length ? index[begin + kr] : -1);
            const bool ok = token >= 0 && int64_t(token) < int64_t(blocks) * page;
            const int chosen = ok ? token : 0;
            if (lane == 0)
                valid[kr] = int(ok);
            const int block = page == 128 ? chosen >> 7 : chosen / page, offset = page == 128 ? chosen & 127 : chosen % page;
            const uint8_t *src = cache + int64_t(block) * bs + int64_t(offset) * ts;
            const auto resource =
                __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0), ok ? (fp4 ? 288 : 528) : 0, 0x00020000);
#endif
                U4 bits;
                if (fp4) {
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
                    const uint32_t data = *reinterpret_cast<const uint32_t *>(packed + col / 2);
                    const uint32_t sf = *reinterpret_cast<const uint32_t *>(packed_scale + ((col / 16) & ~3));
#else
                const uint32_t data = __builtin_amdgcn_raw_buffer_load_b32(resource, col / 2, 0, 0);
                const uint32_t sf = __builtin_amdgcn_raw_buffer_load_b32(resource, 256 + ((col / 16) & ~3), 0, 0);
#endif
#ifdef FLASH_MLA_AMD_CACHE_TABLE
                    bits = decode_fp4_table(data, uint8_t(sf >> ((col / 16 % 4) * 8)));
#else
                const float scale = e4(uint8_t(sf >> ((col / 16 % 4) * 8)));
#if defined(FLASH_MLA_AMD_FP4_HALF_PAIR) || defined(FLASH_MLA_AMD_FP4_FLOAT_PAIR)
                bits = U4{decode_fp4_pair<0>(data, scale), decode_fp4_pair<1>(data, scale), decode_fp4_pair<2>(data, scale),
                    decode_fp4_pair<3>(data, scale)};
#else
                bits[0] = fp4_pair<0>(data, scale);
                bits[1] = fp4_pair<1>(data, scale);
                bits[2] = fp4_pair<2>(data, scale);
                bits[3] = fp4_pair<3>(data, scale);
#endif
#endif
                } else {
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
                    const uint64_t data = *reinterpret_cast<const uint64_t *>(packed + col);
                    const uint32_t sf = *reinterpret_cast<const uint32_t *>(packed_scale + ((col / 32) & ~3));
#else
                const uint32_t a = __builtin_amdgcn_raw_buffer_load_b32(resource, col, 0, 0),
                               b = __builtin_amdgcn_raw_buffer_load_b32(resource, col + 4, 0, 0);
                const uint64_t data = uint64_t(a) | (uint64_t(b) << 32);
                const uint32_t sf = __builtin_amdgcn_raw_buffer_load_b32(resource, 512 + ((col / 32) & ~3), 0, 0);
#endif
#ifdef FLASH_MLA_AMD_NATIVE_FP8_CACHE
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        bits[j] = decode_fp8_pair(uint32_t(data >> (j * 16)), uint8_t(sf >> ((col / 32 % 4) * 8)));
#else
                const float scale = ue8(uint8_t(sf >> ((col / 32 % 4) * 8)));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const auto v = __builtin_amdgcn_cvt_pk_f32_fp8(uint32_t(data >> (j * 16)), false);
                    bits[j] = pack2(v[0] * scale, v[1] * scale);
                }
#endif
                }
                *reinterpret_cast<U4 *>(kv + kpos(kr, col)) = bits;
            }
#endif
        sync_current_lds();
#ifdef FLASH_MLA_AMD_PRODUCE_EXTRA_PACKET
        if constexpr (FIXED == 6 || FIXED == 7) {
            if (extra) {
                constexpr int Body = FIXED == 6 ? 288 : 1024;
                constexpr int Bytes = BK * (Body + 4);
                auto *dst = packet_out + (int64_t(row) * ((p.extra_topk + BK - 1) / BK) + tile - mt) * Bytes;
                if constexpr (FIXED == 6) {
                    // Validity and packed planes are already available in raw LDS.
                    // Finish every wave's reads before prefetch can recycle that tile.
                    for (int i = tid * 16; i < Bytes; i += T * 16)
                        *reinterpret_cast<U4 *>(dst + i) = *reinterpret_cast<const U4 *>(raw + i);
                    sync_current_lds();
                } else {
                    // Decoded LDS is kept live until the end-of-tile matrix join.
                    for (int kr = wave; kr < BK; kr += W)
                        *reinterpret_cast<U4 *>(dst + kr * 1024 + lane * 16) =
                            *reinterpret_cast<const U4 *>(kv + physical_kpos(kr, lane * 8));
                    if (wave == 0)
                        reinterpret_cast<int *>(dst + BK * Body)[lane] = valid[lane];
                }
            }
        }
#endif
        FLASH_MLA_AMD_COMPACT_MARK(3 + tile * 8);
#if defined(FLASH_MLA_AMD_COMPACT_PIPELINE) && !defined(FLASH_MLA_AMD_EARLY_KV_PREFETCH)
        if (tile + 1 < mt + et && (!decoded_swa || tile + 1 >= mt)) {
            prefetch(raw, tile + 1);
            FLASH_MLA_AMD_COMPACT_MARK(10 + tile * 8);
        }
#endif
#endif
        F4 s = {};
        if (wave < QKW) {
#ifdef FLASH_MLA_AMD_COMPACT_CHAINS
            F4 secondary = {};
#ifdef FLASH_MLA_AMD_COMPACT_FOUR_CHAINS
            F4 third = {}, fourth = {};
#endif
#endif
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                I4 a;
#ifdef FLASH_MLA_AMD_REGISTER_QUERY
                a = k < 8 ? low[k] : high[k - 8];
#else
                if (k < 8)
                    a = lds_vector((const __attribute__((address_space(3))) I4 *)(q + qpos(qh, k * 32 + g * 8)));
                else
                    a = high[k - 8];
#endif
                const I4 b = lds_vector((const __attribute__((address_space(3))) I4 *)(kv + kpos(kh, k * 32 + g * 8)));
#ifdef FLASH_MLA_AMD_COMPACT_CHAINS
#ifdef FLASH_MLA_AMD_COMPACT_FOUR_CHAINS
                if (k % 4 == 3)
                    fourth = mma(b, a, fourth);
                else if (k % 4 == 2)
                    third = mma(b, a, third);
                else if (k % 4 == 1)
                    secondary = mma(b, a, secondary);
                else
                    s = mma(b, a, s);
#else
                if (k % 2)
                    secondary = mma(b, a, secondary);
                else
                    s = mma(b, a, s);
#endif
#else
                s = mma(b, a, s);
#endif
                if (k % 4 == 3)
                    __builtin_amdgcn_sched_barrier(0);
            }
#ifdef FLASH_MLA_AMD_COMPACT_CHAINS
#ifdef FLASH_MLA_AMD_COMPACT_FOUR_CHAINS
            s = (s + secondary) + (third + fourth);
#else
            s += secondary;
#endif
#endif
            const int keybase = qw % (BK / 16) * 16 + g * 4;
#ifdef FLASH_MLA_AMD_FULL_VALID_TILE
            if (__builtin_amdgcn_readfirstlane(valid[0]) == 2) {
                s *= p.softmax_scale;
            } else
#endif
            {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    s[j] = valid[keybase + j] ? s[j] * p.softmax_scale : -INFINITY;
            }
#ifdef FLASH_MLA_AMD_COMPACT_REGISTER_SCORE
            float tile_max = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16, 64));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 32, 64));
            if (g == 0)
                reductions[qh * 8 + wave % PARTS] = tile_max;
#ifdef FLASH_MLA_AMD_COMPACT_RELOAD_SCORE
            if (wave % PARTS >= 2)
                *reinterpret_cast<F4 *>(score + spos<32>(qh, keybase - 32)) = s;
#endif
#else
            *reinterpret_cast<F4 *>(score + spos<BK>(qh, keybase)) = s;
#endif
        }
        sync_current_lds();
#ifdef FLASH_MLA_AMD_OVERLAP_DECODE
        // Reader waves have consumed the next packed plane while matrix
        // waves computed QK. All waves can now replace the raw stage.
        if (tile + 2 < mt + et)
            prefetch(raw, tile + 2);
#endif
        FLASH_MLA_AMD_COMPACT_MARK(4 + tile * 8);
#ifdef FLASH_MLA_AMD_TWO_SOFTMAX
        if constexpr (BK == 64) {
            const bool second = whole || begin + 32 < length;
            auto *alpha_second = state + H * 3;
            for (int h = tid / 16; h < H; h += T / 16) {
                const int key = (tid % 16) * 2;
#ifdef FLASH_MLA_AMD_RUNNING_ROW_STATE
                // The same 16 lanes own this head across every 64-key tile.
                static_assert(T / 16 == H);
                float maximum = running_max, denominator = running_sum;
#else
                float maximum = state[h * 3], denominator = state[h * 3 + 1];
#endif
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    if (part == 1 && !second)
                        break;
                    const float a = score[spos<64>(h, part * 32 + key)], b = score[spos<64>(h, part * 32 + key + 1)];
                    float tile_max = fmaxf(a, b);
                    tile_max = fmaxf(tile_max, xor16<8>(tile_max));
                    tile_max = fmaxf(tile_max, xor16<4>(tile_max));
                    tile_max = fmaxf(tile_max, xor16<2>(tile_max));
                    tile_max = fmaxf(tile_max, xor16<1>(tile_max));
                    const float next = fmaxf(maximum, tile_max), safe = next == -INFINITY ? 0.f : next;
                    const float alpha = ex(maximum - safe), p0 = ex(a - safe), p1 = ex(b - safe);
                    float sum = p0 + p1;
                    sum += xor16<8>(sum);
                    sum += xor16<4>(sum);
                    sum += xor16<2>(sum);
                    sum += xor16<1>(sum);
                    *reinterpret_cast<uint32_t *>(prob + part * H * 32 + psw(h, key)) = pack2(p0, p1);
                    if (tid % 16 == 0) {
                        if (part == 0)
                            state[h * 3 + 2] = alpha;
                        else
                            alpha_second[h] = alpha;
                    }
                    maximum = next;
                    denominator = denominator * alpha + sum;
                }
#ifdef FLASH_MLA_AMD_RUNNING_ROW_STATE
                running_max = maximum;
                running_sum = denominator;
                if (tile + 1 == mt + et && tid % 16 == 0) {
                    state[h * 3] = maximum;
                    state[h * 3 + 1] = denominator;
                }
#else
                if (tid % 16 == 0) {
                    state[h * 3] = maximum;
                    state[h * 3 + 1] = denominator;
                }
#endif
            }
            sync_current_lds();
            const int ph = wave / PARTS * 16 + r, colbase = wave % PARTS * (512 / PARTS);
            using LDVec = const __attribute__((address_space(3))) I4 *;
            using LDScalar = const __attribute__((address_space(3))) float *;
#ifdef FLASH_MLA_AMD_PAIRED_PV
            const I4 a0 = lds_vector((LDVec)(prob + psw(ph, g * 8)));
            const float alpha0 = lds_scalar((LDScalar)(state + ph * 3 + 2));
            const I4 a1 = second ? lds_vector((LDVec)(prob + H * 32 + psw(ph, g * 8))) : I4{};
            const float alpha1 = second ? lds_scalar((LDScalar)(alpha_second + ph)) : 0.f;
#pragma unroll
            for (int j = 0; j < L<H, BK>::OUT; ++j) {
                const I4 b0 = read_v(kv, colbase + j * 16, lane, 0);
                output[j] = mma(b0, a0, output[j] * alpha0);
                if (second)
                    output[j] = mma(read_v(kv, colbase + j * 16, lane, 1), a1, output[j] * alpha1);
                if (j % 4 == 3)
                    __builtin_amdgcn_sched_barrier(0);
            }
#else
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                if (part == 1 && !second)
                    break;
                const I4 a = lds_vector((LDVec)(prob + part * H * 32 + psw(ph, g * 8)));
                const float alpha = lds_scalar((LDScalar)(part ? alpha_second + ph : state + ph * 3 + 2));
#pragma unroll
                for (int j = 0; j < L<H, BK>::OUT; ++j) {
                    output[j] = mma(read_v(kv, colbase + j * 16, lane, part), a, output[j] * alpha);
                    if (j % 4 == 3)
                        __builtin_amdgcn_sched_barrier(0);
                }
            }
#endif
        } else
#endif
        {
            // Loading a wider KV tile must not change the BF16 softmax/PV rounding
            // boundary. Consume it as ordered 32-key subtiles, as in the baseline.
#pragma unroll
            for (int part = 0; part < BK / 32; ++part) {
                if (!whole && begin + part * 32 >= length)
                    continue;
#ifdef FLASH_MLA_AMD_COMPACT_REGISTER_SCORE
                const int keybase = wave % PARTS * 16 + g * 4;
#ifdef FLASH_MLA_AMD_COMPACT_RELOAD_SCORE
                if constexpr (BK == 64)
                    if (part == 1)
                        s = __builtin_bit_cast(
                            F4, lds_vector((const __attribute__((address_space(3))) I4 *)(score + spos<32>(qh, (wave % 2) * 16 + g * 4))));
#endif
                if (wave % PARTS / 2 == part) {
                    const float maximum = fmaxf(reductions[qh * 8 + part * 2], reductions[qh * 8 + part * 2 + 1]);
                    const float next = fmaxf(state[qh * 3], maximum), safe = next == -INFINITY ? 0.f : next;
                    F4 values;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        values[j] = ex(s[j] - safe);
#ifdef FLASH_MLA_AMD_ORDERED_SCORE
                    // The established denominator reduces pairs in XOR 8,4,2,1
                    // order. The first peer is in the other 16-key MFMA wave.
                    auto *pairs = reductions + H * 8 + qh * 16 + (wave % 2) * 8 + 2 * g;
                    pairs[0] = values[0] + values[1];
                    pairs[1] = values[2] + values[3];
#else
                    float sum = (values[0] + values[1]) + (values[2] + values[3]);
                    sum += __shfl_xor(sum, 16, 64);
                    sum += __shfl_xor(sum, 32, 64);
#endif
                    *reinterpret_cast<uint32_t *>(prob + psw(qh, keybase - part * 32)) = pack2(values[0], values[1]);
                    *reinterpret_cast<uint32_t *>(prob + psw(qh, keybase - part * 32 + 2)) = pack2(values[2], values[3]);
                    if (g == 0) {
#ifndef FLASH_MLA_AMD_ORDERED_SCORE
                        reductions[qh * 8 + 4 + wave % PARTS] = sum;
#endif
                        if (wave % 2 == 0) {
                            state[qh * 3 + 2] = ex(state[qh * 3] - safe);
                            state[qh * 3] = next;
                        }
                    }
                }
                sync_current_lds();
                // PV needs alpha, not the denominator. Its leader can update the
                // denominator while other waves consume P, before the PV barrier.
#ifdef FLASH_MLA_AMD_ORDERED_SCORE
                for (int h = tid / 16; h < H; h += T / 16) {
                    float sum = reductions[H * 8 + h * 16 + (tid % 16)];
                    sum += xor16<8>(sum);
                    sum += xor16<4>(sum);
                    sum += xor16<2>(sum);
                    sum += xor16<1>(sum);
                    if (tid % 16 == 0)
                        state[h * 3 + 1] = state[h * 3 + 1] * state[h * 3 + 2] + sum;
                }
#else
                if (tid < H)
                    state[tid * 3 + 1] =
                        state[tid * 3 + 1] * state[tid * 3 + 2] + (reductions[tid * 8 + 4 + part * 2] + reductions[tid * 8 + 5 + part * 2]);
#endif
#else
                for (int h = tid / 16; h < H; h += T / 16) {
                    const int key = (tid % 16) * 2;
                    const float s0 = score[spos<BK>(h, part * 32 + key)], s1 = score[spos<BK>(h, part * 32 + key + 1)];
                    float maximum = fmaxf(s0, s1);
                    maximum = fmaxf(maximum, xor16<8>(maximum));
                    maximum = fmaxf(maximum, xor16<4>(maximum));
                    maximum = fmaxf(maximum, xor16<2>(maximum));
                    maximum = fmaxf(maximum, xor16<1>(maximum));
                    const float next = fmaxf(state[h * 3], maximum), safe = next == -INFINITY ? 0.f : next;
                    const float alpha = ex(state[h * 3] - safe), a = ex(s0 - safe), b = ex(s1 - safe);
                    float sum = a + b;
                    sum += xor16<8>(sum);
                    sum += xor16<4>(sum);
                    sum += xor16<2>(sum);
                    sum += xor16<1>(sum);
                    *reinterpret_cast<uint32_t *>(prob + psw(h, key)) = pack2(a, b);
                    if (tid % 16 == 0) {
                        state[h * 3] = next;
                        state[h * 3 + 1] = state[h * 3 + 1] * alpha + sum;
                        state[h * 3 + 2] = alpha;
                    }
                }
                sync_current_lds();
#endif
                FLASH_MLA_AMD_COMPACT_MARK(5 + tile * 8 + part * 2);
                const int ph = wave / PARTS * 16 + r, colbase = wave % PARTS * (512 / PARTS);
                const I4 a = lds_vector((const __attribute__((address_space(3))) I4 *)(prob + psw(ph, g * 8)));
                const float alpha = lds_scalar((const __attribute__((address_space(3))) float *)(state + ph * 3 + 2));
#pragma unroll
                for (int j = 0; j < L<H, BK>::OUT; ++j) {
                    const I4 b = read_v(kv, colbase + j * 16, lane, part);
                    output[j] = mma(b, a, output[j] * alpha);
#ifdef FLASH_MLA_AMD_COMPACT_REGISTER_SCORE
                    if (j % 2 == 1)
                        __builtin_amdgcn_sched_barrier(0);
#else
                    if (j % 4 == 3)
                        __builtin_amdgcn_sched_barrier(0);
#endif
                }
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
                // The last subtile already meets at the raw-prefetch drain below.
                // Earlier subtiles must finish reading P before it is overwritten.
                if (part + 1 < BK / 32)
                    sync_current_lds();
#else
                sync_current_lds();
#endif
                FLASH_MLA_AMD_COMPACT_MARK(6 + tile * 8 + part * 2);
            }
        }
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
        __builtin_amdgcn_s_waitcnt(0xf70);
        sync_current_lds();
#endif
        FLASH_MLA_AMD_COMPACT_MARK(9 + tile * 8);
#ifdef FLASH_MLA_AMD_SEPARATE_CACHE_LOOPS
    };
    // Main and compressed keys are consumed in their original logical order.
    // A full tile can execute both 32-key online-softmax steps unconditionally.
    for (int i = 0; i < ml / BK; ++i)
        consume(i, std::false_type{}, std::true_type{});
    if (ml % BK)
        consume(mt - 1, std::false_type{}, std::false_type{});
    for (int i = 0; i < el / BK; ++i)
        consume(mt + i, std::true_type{}, std::true_type{});
    if (el % BK)
        consume(mt + et - 1, std::true_type{}, std::false_type{});
#else
    }
#endif
    const int ph = wave / PARTS * 16 + r, colbase = wave % PARTS * (512 / PARTS);
    const float den = state[ph * 3 + 1], inverse = den > 0 ? 1.f / den : 1.f;
    const float logsum = den > 0 ? state[ph * 3] + logf(den) : INFINITY;
    const float sink = den > 0 ? (p.sink ? 1.f / (1.f + ex(p.sink[hb + ph] - logsum)) : 1.f) : 0.f;
    if (p.lse && g == 0 && wave % PARTS == 0)
        p.lse[int64_t(row) * 64 + hb + ph] = logsum;
    // The preceding iteration drain (or the initialization join for no keys)
    // completed every Q/KV reader. State remains in disjoint shared storage;
    // the full-width output can reuse Q/KV without another identical join.
#ifdef FLASH_MLA_AMD_FRAGMENT_OUTPUT
    constexpr int Fragments = L<H, BK>::OUT;
    static_assert(Fragments % 8 == 0);
#pragma unroll
    for (int j = 0; j < Fragments; ++j)
#pragma unroll
        for (int z = 0; z < 4; ++z)
            output[j][z] = bround((output[j][z] * inverse) * sink);
    if (wave % PARTS == PARTS - 1) {
        if constexpr (NEOX) {
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int z = 0; z < 4; ++z) {
                    const int d = 16 * j + g * 4 + z;
                    const float a = output[Fragments - 4 + j][z], b = output[Fragments - 2 + j][z];
                    output[Fragments - 4 + j][z] = bround(fmaf(a, cs[d], b * cs[32 + d]));
                    output[Fragments - 2 + j][z] = bround(fmaf(a, -cs[32 + d], b * cs[d]));
                }
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int z = 0; z < 4; z += 2) {
                    const int d = (16 * j + 4 * g + z) / 2;
                    const float a = output[Fragments - 4 + j][z], b = output[Fragments - 4 + j][z + 1];
                    output[Fragments - 4 + j][z] = bround(fmaf(a, cs[d], b * cs[32 + d]));
                    output[Fragments - 4 + j][z + 1] = bround(fmaf(a, -cs[32 + d], b * cs[d]));
                }
        }
    }
#pragma unroll
    for (int word = 0; word < Fragments / 8; ++word) {
        uint32_t codes = 0;
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int j = word * 8 + group * 2;
            float maximum = 1.e-4f;
#pragma unroll
            for (int k = 0; k < 2; ++k)
#pragma unroll
                for (int z = 0; z < 4; ++z)
                    maximum = fmaxf(maximum, fabsf(output[j + k][z]));
            maximum = fmaxf(maximum, __shfl_xor(maximum, 16, 64));
            maximum = fmaxf(maximum, __shfl_xor(maximum, 32, 64));
            const int exponent = int((__float_as_uint(maximum * (1.f / 448.f)) - 1) >> 23) + 1;
            codes |= uint32_t(exponent) << (group * 8);
            const float scale = __uint_as_float(uint32_t(254 - exponent) << 23);
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const F4 v = output[j + k];
                const int a = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] * scale, v[1] * scale, 0, false);
                const uint32_t bits = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] * scale, v[3] * scale, a, true);
                const int d = colbase + (j + k) * 16 + g * 4;
#ifdef FLASH_MLA_AMD_STAGE_FRAGMENT_OUTPUT
                *reinterpret_cast<uint32_t *>(shared + sw(ph, d)) = bits;
#else
                *reinterpret_cast<uint32_t *>(p.output + int64_t(row) * 32768 + (hb + ph) * 512 + d) = bits;
#endif
            }
        }
        if (g == 0)
            p.scales[row + int64_t((hb + ph) * 4 + colbase / 128 + word) * p.scale_pitch] = codes;
    }
#ifdef FLASH_MLA_AMD_STAGE_FRAGMENT_OUTPUT
    sync_current_lds();
    for (int h = wave; h < H; h += W) {
        const uint64_t bits = *reinterpret_cast<const uint64_t *>(shared + sw(h, lane * 8));
        *reinterpret_cast<uint64_t *>(p.output + int64_t(row) * 32768 + (hb + h) * 512 + lane * 8) = bits;
    }
#endif
#else
#pragma unroll
    for (int j = 0; j < L<H, BK>::OUT; ++j) {
        const int d = colbase + j * 16 + g * 4;
        const uint32_t a = pack2((output[j][0] * inverse) * sink, (output[j][1] * inverse) * sink);
        const uint32_t b = pack2((output[j][2] * inverse) * sink, (output[j][3] * inverse) * sink);
        *reinterpret_cast<uint64_t *>(q + sw(ph, d)) = uint64_t(a) | (uint64_t(b) << 32);
    }
    sync_current_lds();
    for (int h = wave; h < H; h += W) {
        const int col = lane * 8;
        const U4 raw = *reinterpret_cast<const U4 *>(q + sw(h, col));
        float v[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j * 2] = bf(raw[j]);
            v[j * 2 + 1] = bf(raw[j] >> 16);
        }
        rope<NEOX>(v, cs, col, lane, true);
        float maximum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = bround(v[j]);
            maximum = fmaxf(maximum, fabsf(v[j]));
        }
        maximum = fmaxf(maximum, __shfl_xor(maximum, 1, 64));
        maximum = fmaxf(maximum, __shfl_xor(maximum, 2, 64));
        const uint32_t bits = __float_as_uint(fmaxf(maximum, 1e-4f) * (1.f / 448.f));
        const int exponent = int((bits - 1) >> 23) + 1;
        const float scale = __uint_as_float(uint32_t(254 - exponent) << 23);
        const int a = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] * scale, v[1] * scale, 0, false),
                  b = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] * scale, v[5] * scale, 0, false);
        const uint32_t x = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] * scale, v[3] * scale, a, true),
                       y = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] * scale, v[7] * scale, b, true);
        *reinterpret_cast<uint64_t *>(p.output + int64_t(row) * 32768 + (hb + h) * 512 + col) = uint64_t(x) | (uint64_t(y) << 32);
        uint32_t sf = exponent;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 4, 64)) << 8;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 8, 64)) << 16;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 12, 64)) << 24;
        if (lane % 16 == 0)
            p.scales[row + int64_t((hb + h) * 4 + lane / 16) * p.scale_pitch] = sf;
    }
#endif
    FLASH_MLA_AMD_COMPACT_MARK(2 + (mt + et) * 8);
}

inline bool valid_args(const Args *p) {
    if (!p || !p->q || !p->cache || !p->indices || !p->positions || !p->cos_sin || !p->output || !p->scales || p->rows < 1 ||
        p->rows > 1536 || p->q_stride < 32768 || p->page_size < 1 || p->cache_blocks < 1 || p->topk < 0 || p->topk > 640 ||
        p->extra_topk < 0 || p->extra_topk > 512 || p->index_stride < p->topk || p->scale_pitch < p->rows || p->cos_sin_stride < 64 ||
        (p->positions64 != 0 && p->positions64 != 1) || (p->neox != 0 && p->neox != 1) ||
        (p->already_roped != 0 && p->already_roped != 1) || !std::isfinite(p->softmax_scale) || p->softmax_scale <= 0)
        return false;
    return !p->extra_topk || (p->extra && p->extra_indices && p->extra_blocks > 0 && p->extra_page_size > 0 &&
                                 p->extra_index_stride >= p->extra_topk && (p->extra_format == 4 || p->extra_format == 8));
}
template <int H, int BK, bool N, bool I, bool C = false, int F = 0> int prepare() {
    return hipFuncSetAttribute((const void *)kernel<H, BK, N, I, C, F>, hipFuncAttributeMaxDynamicSharedMemorySize, L<H, BK>::BYTES);
}
template <int H, int BK, bool N, bool I, bool C = false, int F = 0> int launch(Args p, hipStream_t s, uint8_t *out = nullptr) {
    hipLaunchKernelGGL((kernel<H, BK, N, I, C, F>), dim3(p.rows, 64 / H), dim3(L<H, BK>::T), (L<H, BK>::BYTES), s, p, out);
    return hipGetLastError();
}
template <int BK> int prepare_all(int h) {
    int status = 0;
#ifdef FLASH_MLA_AMD_STRIDED_PAGES
    if constexpr (BK == 64)
        if (h == 64) {
#define PAGE_PREP(F)                                                                                                                       \
    status = prepare<64, 64, false, false, false, F>();                                                                                    \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<64, 64, false, true, false, F>();                                                                                     \
    if (status)                                                                                                                            \
    return status
            PAGE_PREP(1);
            PAGE_PREP(2);
            PAGE_PREP(3);
#undef PAGE_PREP
        }
#endif
#ifdef FLASH_MLA_AMD_COMPACT_STATIC_LAYOUT
#define PREP(H)                                                                                                                            \
    status = prepare<H, BK, false, false, true>();                                                                                         \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<H, BK, false, true, true>();                                                                                          \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<H, BK, true, false, true>();                                                                                          \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<H, BK, true, true, true>();                                                                                           \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<H, BK, false, false>();                                                                                               \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<H, BK, false, true>();                                                                                                \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<H, BK, true, false>();                                                                                                \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    return prepare<H, BK, true, true>();
#else
#define PREP(H)                                                                                                                            \
    status = prepare<H, BK, false, false>();                                                                                               \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<H, BK, false, true>();                                                                                                \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    status = prepare<H, BK, true, false>();                                                                                                \
    if (status)                                                                                                                            \
        return status;                                                                                                                     \
    return prepare<H, BK, true, true>();
#endif
    if (h == 32) {
        PREP(32)
    }
    if (h == 64) {
        PREP(64)
    }
#undef PREP
    return hipErrorInvalidValue;
}
template <int BK> int launch_all(const Args *p, int h, void *stream) {
    if (!valid_args(p))
        return hipErrorInvalidValue;
    auto s = static_cast<hipStream_t>(stream);
#ifdef FLASH_MLA_AMD_STRIDED_PAGES
    if constexpr (BK == 64) {
        if (h == 64 && !p->neox && p->page_size == 133 && p->cache_token_stride == 528 && p->cache_block_stride == 3094080 &&
            uint64_t(p->cache_blocks) * 5860 <= UINT32_MAX) {
#define PAGE_RUN(F) (p->positions64 ? launch<64, 64, false, true, false, F>(*p, s) : launch<64, 64, false, false, false, F>(*p, s))
            if (p->rows == 500 && p->topk == 128) {
                if (!p->extra_topk)
                    return PAGE_RUN(1);
                if (p->extra_topk == 512 && p->extra_format == 4 && p->extra_page_size == 64 && p->extra_token_stride == 288 &&
                    p->extra_block_stride == 114048 && uint64_t(p->extra_blocks) * 396 <= UINT32_MAX)
                    return PAGE_RUN(2);
            } else if (p->rows == 400 && p->topk == 132 && !p->extra_topk)
                return PAGE_RUN(3);
#undef PAGE_RUN
        }
    }
#endif
#ifdef FLASH_MLA_AMD_COMPACT_STATIC_LAYOUT
    const int extra_bytes = p->extra_format == 4 ? 288 : 528;
    const bool contiguous =
        p->cache_token_stride == 528 && p->cache_block_stride == int64_t(p->page_size) * 528 &&
        (!p->extra_topk || (p->extra_token_stride == extra_bytes && p->extra_block_stride == int64_t(p->extra_page_size) * extra_bytes));
#define RUN(H)                                                                                                                             \
    if (contiguous) {                                                                                                                      \
        if (p->neox)                                                                                                                       \
            return p->positions64 ? launch<H, BK, true, true, true>(*p, s) : launch<H, BK, true, false, true>(*p, s);                      \
        return p->positions64 ? launch<H, BK, false, true, true>(*p, s) : launch<H, BK, false, false, true>(*p, s);                        \
    }                                                                                                                                      \
    if (p->neox)                                                                                                                           \
        return p->positions64 ? launch<H, BK, true, true>(*p, s) : launch<H, BK, true, false>(*p, s);                                      \
    return p->positions64 ? launch<H, BK, false, true>(*p, s) : launch<H, BK, false, false>(*p, s);
#else
#define RUN(H)                                                                                                                             \
    if (p->neox)                                                                                                                           \
        return p->positions64 ? launch<H, BK, true, true>(*p, s) : launch<H, BK, true, false>(*p, s);                                      \
    return p->positions64 ? launch<H, BK, false, true>(*p, s) : launch<H, BK, false, false>(*p, s);
#endif
    if (h == 32) {
        RUN(32)
    }
    if (h == 64) {
        RUN(64)
    }
#undef RUN
    return hipErrorInvalidValue;
}
}
#ifdef FLASH_MLA_AMD_OVERLAP_DECODE
extern "C" int fused_attention_overlap32_prepare(int h) { return flash_mla::amd::compact_attention::prepare_all<32>(h); }
extern "C" int fused_attention_overlap32_launch(const flash_mla::amd::fused_sparse::Args *p, int h, void *s) {
    return flash_mla::amd::compact_attention::launch_all<32>(p, h, s);
}
#else
extern "C" int fused_attention_compact32_prepare(int h) { return flash_mla::amd::compact_attention::prepare_all<32>(h); }
extern "C" int fused_attention_compact64_prepare(int h) { return flash_mla::amd::compact_attention::prepare_all<64>(h); }
extern "C" int fused_attention_compact32_launch(const flash_mla::amd::fused_sparse::Args *p, int h, void *s) {
    return flash_mla::amd::compact_attention::launch_all<32>(p, h, s);
}
extern "C" int fused_attention_compact64_launch(const flash_mla::amd::fused_sparse::Args *p, int h, void *s) {
    return flash_mla::amd::compact_attention::launch_all<64>(p, h, s);
}
#endif
