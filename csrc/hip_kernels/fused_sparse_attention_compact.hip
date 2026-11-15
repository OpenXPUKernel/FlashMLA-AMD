// Native AMD GPU compact attention. BF16/FP32 arithmetic and packed cache I/O.
// Half of Q stays in registers; padded LDS rows allow immediate-offset reads.
#include "fused_sparse_attention.hip"
#ifndef FLASH_MLA_AMD_COMPACT_MARK
#define FLASH_MLA_AMD_COMPACT_MARK(index)                                                                              \
    do {                                                                                                               \
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
    static constexpr int Q = 0, KV = H * 272 * 2, SCORE = KV + BK * KSTRIDE * 2, PROB = SCORE + H * BK * 4;
    static constexpr int STATE = PROB + H * 32 * 2, VALID = STATE + H * 3 * 4;
    static constexpr int RAW = (VALID + BK * 4 + 255) / 256 * 256;
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
    static constexpr int BYTES = RAW + BK * 544;
#else
    static constexpr int BYTES = RAW;
#endif
    static_assert(BYTES >= H * 512 * 2, "the final output reuses Q/KV storage");
};
__device__ __forceinline__ int qpos(int row, int col) { return row * 272 + col; }
__device__ __forceinline__ int physical_row(int row) {
    return (row & ~31) | (row & 3) | (((row >> 3) & 3) << 2) | (((row >> 2) & 1) << 4);
}
__device__ __forceinline__ int physical_kpos(int row, int col) {
#ifdef FLASH_MLA_AMD_COMPACT_KV_SWIZZLE
    col ^= ((row & 3) << 4) ^ (((row >> 3) & 1) << 6);
#endif
    return row * KSTRIDE + col;
}
__device__ __forceinline__ int kpos(int row, int col) { return physical_kpos(physical_row(row), col); }
template <int BK> __device__ __forceinline__ int spos(int row, int col) { return row * BK + (col ^ ((row & 7) << 2)); }
__device__ __forceinline__ I4 lds_vector(const __attribute__((address_space(3))) I4 *__restrict__ p) { return *p; }
__device__ __forceinline__ float lds_scalar(const __attribute__((address_space(3))) float *__restrict__ p) {
    return *p;
}
__device__ __forceinline__ I2 lds_transpose(const __attribute__((address_space(3))) S4 *__restrict__ p) {
    return __builtin_bit_cast(
        I2, __builtin_amdgcn_ds_read_tr16_b64_v4i16(const_cast<__attribute__((address_space(3))) S4 *>(p)));
}
template <int CTRL> __device__ __forceinline__ float dpp(float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), CTRL, 0xf, 0xf, false));
}
template <int MASK> __device__ __forceinline__ float xor16(float x) {
    if constexpr (MASK == 8)
        return dpp<0x141>(dpp<0x140>(x));
    if constexpr (MASK == 4)
        return dpp<0x1b>(dpp<0x141>(x));
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
template <int H, int BK, int W = L<H, BK>::W, bool CONTIG = false>
__device__ __forceinline__ void prefetch_raw(const Args &p, uint8_t *raw, int row, int tile, int mt, int ml, int el,
                                             int wave, int lane) {
    const bool extra = tile >= mt;
    const int begin = (extra ? tile - mt : tile) * BK, length = extra ? el : ml;
    const int32_t *index =
        (extra ? p.extra_indices : p.indices) + int64_t(row) * (extra ? p.extra_index_stride : p.index_stride);
    const uint8_t *cache = extra ? p.extra : p.cache;
    const int page = extra ? p.extra_page_size : p.page_size, blocks = extra ? p.extra_blocks : p.cache_blocks;
    const int64_t bs = extra ? p.extra_block_stride : p.cache_block_stride,
                  ts = extra ? p.extra_token_stride : p.cache_token_stride;
    const int bytes = extra && p.extra_format == 4 ? 288 : 528;
#ifdef FLASH_MLA_AMD_COMPACT_LINEAR_ADDRESS
    const bool contiguous = ts == bytes && bs == int64_t(page) * bytes;
#endif
    // Fetch this tile's indices before issuing any asynchronous row transfer.
    // Otherwise the next index load can force completion of the previous DMA.
    int fetched = lane < BK && begin + lane < length ? index[begin + lane] : -1;
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
                const int block = page == 128 ? chosen >> 7 : chosen / page,
                          offset = page == 128 ? chosen & 127 : chosen % page;
                src = cache + int64_t(block) * bs + int64_t(offset) * ts;
#ifdef FLASH_MLA_AMD_COMPACT_LINEAR_ADDRESS
            }
#endif
        }
#endif
        const auto resource =
            __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0), ok ? bytes : 0, 0x00020000);
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

template <int H, int BK, bool NEOX, bool POS64, bool CONTIG = false>
__global__ __launch_bounds__((L<H, BK>::T)) void kernel(Args p) {
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
    const int tid = threadIdx.x, lane = tid % 64, r = lane % 16, g = lane / 16,
              wave = __builtin_amdgcn_readfirstlane(tid / 64);
    constexpr int QKW = H / 16 * (BK / 16);
    const int qw = wave % QKW;
    const int row = blockIdx.x, hb = blockIdx.y * H, qh = qw / (BK / 16) * 16 + r, kh = qw % (BK / 16) * 16 + r;
    FLASH_MLA_AMD_COMPACT_MARK(0);
    const int64_t pos =
        POS64 ? static_cast<const int64_t *>(p.positions)[row] : static_cast<const int32_t *>(p.positions)[row];
    const float *cs = p.cos_sin + pos * p.cos_sin_stride;
    if (p.invocations && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        atomicAdd(p.invocations, 1ULL);
    for (int h = wave; h < H; h += W) {
        if (lane < 32)
            *reinterpret_cast<U4 *>(q + qpos(h, lane * 8)) =
                *reinterpret_cast<const U4 *>(p.q + int64_t(row) * p.q_stride + (hb + h) * 512 + lane * 8);
    }
    I4 high[8];
    if (wave < QKW) {
#pragma unroll
        for (int k = 0; k < 8; ++k)
            high[k] =
                *reinterpret_cast<const I4 *>(p.q + int64_t(row) * p.q_stride + (hb + qh) * 512 + 256 + k * 32 + g * 8);
        if (!p.already_roped)
            rotate_tail<NEOX>(high[6], high[7], cs, g);
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
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
    FLASH_MLA_AMD_COMPACT_MARK(1);
    F4 output[L<H, BK>::OUT] = {};
    const int ml = p.lengths ? min(max(p.lengths[row], 0), p.topk) : p.topk;
    const int el = p.extra_lengths ? min(max(p.extra_lengths[row], 0), p.extra_topk) : p.extra_topk;
    const int mt = (ml + BK - 1) / BK, et = (el + BK - 1) / BK;
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
    if (mt + et) {
        prefetch_raw<H, BK, L<H, BK>::W, CONTIG>(p, raw, row, 0, mt, ml, el, wave, lane);
        __builtin_amdgcn_s_waitcnt(0xf70);
        sync_current_lds();
    }
#endif
    FLASH_MLA_AMD_COMPACT_MARK(2);
    for (int tile = 0; tile < mt + et; ++tile) {
        const bool extra = tile >= mt;
        const int begin = (extra ? tile - mt : tile) * BK, length = extra ? el : ml;
        const int32_t *index =
            (extra ? p.extra_indices : p.indices) + int64_t(row) * (extra ? p.extra_index_stride : p.index_stride);
        const uint8_t *cache = extra ? p.extra : p.cache;
        const int page = extra ? p.extra_page_size : p.page_size, blocks = extra ? p.extra_blocks : p.cache_blocks;
        const int64_t bs = extra ? p.extra_block_stride : p.cache_block_stride,
                      ts = extra ? p.extra_token_stride : p.cache_token_stride;
        const bool fp4 = extra && p.extra_format == 4;
#ifdef FLASH_MLA_AMD_COMPACT_UNROLL_ROWS
#pragma unroll
#else
#pragma unroll 2
#endif
        for (int b = 0; b < BK / W; ++b) {
            const int kr = wave + b * W, col = lane * 8;
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
            const uint8_t *packed = raw + kr * 544;
            if (lane == 0)
                valid[kr] = *reinterpret_cast<const int *>(packed + 540);
#else
            const int token = __builtin_amdgcn_readfirstlane(begin + kr < length ? index[begin + kr] : -1);
            const bool ok = token >= 0 && int64_t(token) < int64_t(blocks) * page;
            const int chosen = ok ? token : 0;
            if (lane == 0)
                valid[kr] = int(ok);
            const int block = page == 128 ? chosen >> 7 : chosen / page,
                      offset = page == 128 ? chosen & 127 : chosen % page;
            const uint8_t *src = cache + int64_t(block) * bs + int64_t(offset) * ts;
            const auto resource = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0),
                                                                    ok ? (fp4 ? 288 : 528) : 0, 0x00020000);
#endif
            U4 bits;
            if (fp4) {
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
                const uint32_t data = *reinterpret_cast<const uint32_t *>(packed + col / 2);
                const uint32_t sf = *reinterpret_cast<const uint32_t *>(packed + 256 + ((col / 16) & ~3));
#else
                const uint32_t data = __builtin_amdgcn_raw_buffer_load_b32(resource, col / 2, 0, 0);
                const uint32_t sf = __builtin_amdgcn_raw_buffer_load_b32(resource, 256 + ((col / 16) & ~3), 0, 0);
#endif
                const float scale = e4(uint8_t(sf >> ((col / 16 % 4) * 8)));
                bits[0] = fp4_pair<0>(data, scale);
                bits[1] = fp4_pair<1>(data, scale);
                bits[2] = fp4_pair<2>(data, scale);
                bits[3] = fp4_pair<3>(data, scale);
            } else {
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
                const uint64_t data = *reinterpret_cast<const uint64_t *>(packed + col);
                const uint32_t sf = *reinterpret_cast<const uint32_t *>(packed + 512 + ((col / 32) & ~3));
#else
                const uint32_t a = __builtin_amdgcn_raw_buffer_load_b32(resource, col, 0, 0),
                               b = __builtin_amdgcn_raw_buffer_load_b32(resource, col + 4, 0, 0);
                const uint64_t data = uint64_t(a) | (uint64_t(b) << 32);
                const uint32_t sf = __builtin_amdgcn_raw_buffer_load_b32(resource, 512 + ((col / 32) & ~3), 0, 0);
#endif
                const float scale = ue8(uint8_t(sf >> ((col / 32 % 4) * 8)));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const auto v = __builtin_amdgcn_cvt_pk_f32_fp8(uint32_t(data >> (j * 16)), false);
                    bits[j] = pack2(v[0] * scale, v[1] * scale);
                }
            }
            *reinterpret_cast<U4 *>(kv + kpos(kr, col)) = bits;
        }
        sync_current_lds();
        FLASH_MLA_AMD_COMPACT_MARK(3 + tile * 8);
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
        if (tile + 1 < mt + et) {
            prefetch_raw<H, BK, L<H, BK>::W, CONTIG>(p, raw, row, tile + 1, mt, ml, el, wave, lane);
            FLASH_MLA_AMD_COMPACT_MARK(10 + tile * 8);
        }
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
                if (k < 8)
                    a = lds_vector((const __attribute__((address_space(3))) I4 *)(q + qpos(qh, k * 32 + g * 8)));
                else
                    a = high[k - 8];
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
#pragma unroll
            for (int j = 0; j < 4; ++j)
                s[j] = valid[keybase + j] ? s[j] * p.softmax_scale : -INFINITY;
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
        FLASH_MLA_AMD_COMPACT_MARK(4 + tile * 8);
        // Loading a wider KV tile must not change the BF16 softmax/PV rounding
        // boundary. Consume it as ordered 32-key subtiles, as in the baseline.
#pragma unroll
        for (int part = 0; part < BK / 32; ++part) {
            if (begin + part * 32 >= length)
                continue;
#ifdef FLASH_MLA_AMD_COMPACT_REGISTER_SCORE
            const int keybase = wave % PARTS * 16 + g * 4;
#ifdef FLASH_MLA_AMD_COMPACT_RELOAD_SCORE
            if constexpr (BK == 64)
                if (part == 1)
                    s = __builtin_bit_cast(F4, lds_vector((const __attribute__((address_space(3)))
                                                           I4 *)(score + spos<32>(qh, (wave % 2) * 16 + g * 4))));
#endif
            if (wave % PARTS / 2 == part) {
                const float maximum = fmaxf(reductions[qh * 8 + part * 2], reductions[qh * 8 + part * 2 + 1]);
                const float next = fmaxf(state[qh * 3], maximum), safe = next == -INFINITY ? 0.f : next;
                F4 values;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    values[j] = ex(s[j] - safe);
                float sum = (values[0] + values[1]) + (values[2] + values[3]);
                sum += __shfl_xor(sum, 16, 64);
                sum += __shfl_xor(sum, 32, 64);
                *reinterpret_cast<uint32_t *>(prob + psw(qh, keybase - part * 32)) = pack2(values[0], values[1]);
                *reinterpret_cast<uint32_t *>(prob + psw(qh, keybase - part * 32 + 2)) = pack2(values[2], values[3]);
                if (g == 0) {
                    reductions[qh * 8 + 4 + wave % PARTS] = sum;
                    if (wave % 2 == 0) {
                        state[qh * 3 + 2] = ex(state[qh * 3] - safe);
                        state[qh * 3] = next;
                    }
                }
            }
            sync_current_lds();
            // PV needs alpha, not the denominator. Its leader can update the
            // denominator while other waves consume P, before the PV barrier.
            if (tid < H)
                state[tid * 3 + 1] = state[tid * 3 + 1] * state[tid * 3 + 2] +
                                     (reductions[tid * 8 + 4 + part * 2] + reductions[tid * 8 + 5 + part * 2]);
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
            sync_current_lds();
            FLASH_MLA_AMD_COMPACT_MARK(6 + tile * 8 + part * 2);
        }
#ifdef FLASH_MLA_AMD_COMPACT_PIPELINE
        __builtin_amdgcn_s_waitcnt(0xf70);
        sync_current_lds();
#endif
        FLASH_MLA_AMD_COMPACT_MARK(9 + tile * 8);
    }
    const int ph = wave / PARTS * 16 + r, colbase = wave % PARTS * (512 / PARTS);
    const float den = state[ph * 3 + 1], inverse = den > 0 ? 1.f / den : 1.f;
    const float logsum = den > 0 ? state[ph * 3] + logf(den) : INFINITY;
    const float sink = den > 0 ? (p.sink ? 1.f / (1.f + ex(p.sink[hb + ph] - logsum)) : 1.f) : 0.f;
    if (p.lse && g == 0 && wave % PARTS == 0)
        p.lse[int64_t(row) * 64 + hb + ph] = logsum;
    // The full-width final output now owns the completed Q/KV workspace.
    sync_current_lds();
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
        *reinterpret_cast<uint64_t *>(p.output + int64_t(row) * 32768 + (hb + h) * 512 + col) =
            uint64_t(x) | (uint64_t(y) << 32);
        uint32_t sf = exponent;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 4, 64)) << 8;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 8, 64)) << 16;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 12, 64)) << 24;
        if (lane % 16 == 0)
            p.scales[row + int64_t((hb + h) * 4 + lane / 16) * p.scale_pitch] = sf;
    }
    FLASH_MLA_AMD_COMPACT_MARK(2 + (mt + et) * 8);
}

inline bool valid_args(const Args *p) {
    if (!p || !p->q || !p->cache || !p->indices || !p->positions || !p->cos_sin || !p->output || !p->scales ||
        p->rows < 1 || p->rows > 1536 || p->q_stride < 32768 || p->page_size < 1 || p->cache_blocks < 1 ||
        p->topk < 0 || p->topk > 640 || p->extra_topk < 0 || p->extra_topk > 512 || p->index_stride < p->topk ||
        p->scale_pitch < p->rows || p->cos_sin_stride < 64 || (p->positions64 != 0 && p->positions64 != 1) ||
        (p->neox != 0 && p->neox != 1) || (p->already_roped != 0 && p->already_roped != 1) ||
        !std::isfinite(p->softmax_scale) || p->softmax_scale <= 0)
        return false;
    return !p->extra_topk || (p->extra && p->extra_indices && p->extra_blocks > 0 && p->extra_page_size > 0 &&
                              p->extra_index_stride >= p->extra_topk && (p->extra_format == 4 || p->extra_format == 8));
}
template <int H, int BK, bool N, bool I, bool C = false> int prepare() {
    return hipFuncSetAttribute((const void *)kernel<H, BK, N, I, C>, hipFuncAttributeMaxDynamicSharedMemorySize,
                               L<H, BK>::BYTES);
}
template <int H, int BK, bool N, bool I, bool C = false> int launch(Args p, hipStream_t s) {
    hipLaunchKernelGGL((kernel<H, BK, N, I, C>), dim3(p.rows, 64 / H), dim3(L<H, BK>::T), (L<H, BK>::BYTES), s, p);
    return hipGetLastError();
}
template <int BK> int prepare_all(int h) {
    int status = 0;
#ifdef FLASH_MLA_AMD_COMPACT_STATIC_LAYOUT
#define PREP(H)                                                                                                        \
    status = prepare<H, BK, false, false, true>();                                                                     \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, BK, false, true, true>();                                                                      \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, BK, true, false, true>();                                                                      \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, BK, true, true, true>();                                                                       \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, BK, false, false>();                                                                           \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, BK, false, true>();                                                                            \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, BK, true, false>();                                                                            \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    return prepare<H, BK, true, true>();
#else
#define PREP(H)                                                                                                        \
    status = prepare<H, BK, false, false>();                                                                           \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, BK, false, true>();                                                                            \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, BK, true, false>();                                                                            \
    if (status)                                                                                                        \
        return status;                                                                                                 \
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
#ifdef FLASH_MLA_AMD_COMPACT_STATIC_LAYOUT
    const int extra_bytes = p->extra_format == 4 ? 288 : 528;
    const bool contiguous = p->cache_token_stride == 528 && p->cache_block_stride == int64_t(p->page_size) * 528 &&
                            (!p->extra_topk || (p->extra_token_stride == extra_bytes &&
                                                p->extra_block_stride == int64_t(p->extra_page_size) * extra_bytes));
#define RUN(H)                                                                                                         \
    if (contiguous) {                                                                                                  \
        if (p->neox)                                                                                                   \
            return p->positions64 ? launch<H, BK, true, true, true>(*p, s) : launch<H, BK, true, false, true>(*p, s);  \
        return p->positions64 ? launch<H, BK, false, true, true>(*p, s) : launch<H, BK, false, false, true>(*p, s);    \
    }                                                                                                                  \
    if (p->neox)                                                                                                       \
        return p->positions64 ? launch<H, BK, true, true>(*p, s) : launch<H, BK, true, false>(*p, s);                  \
    return p->positions64 ? launch<H, BK, false, true>(*p, s) : launch<H, BK, false, false>(*p, s);
#else
#define RUN(H)                                                                                                         \
    if (p->neox)                                                                                                       \
        return p->positions64 ? launch<H, BK, true, true>(*p, s) : launch<H, BK, true, false>(*p, s);                  \
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
} // namespace flash_mla::amd::compact_attention
extern "C" int fused_attention_compact32_prepare(int h) {
    return flash_mla::amd::compact_attention::prepare_all<32>(h);
}
extern "C" int fused_attention_compact64_prepare(int h) {
    return flash_mla::amd::compact_attention::prepare_all<64>(h);
}
extern "C" int fused_attention_compact32_launch(const flash_mla::amd::fused_sparse::Args *p, int h, void *s) {
    return flash_mla::amd::compact_attention::launch_all<32>(p, h, s);
}
extern "C" int fused_attention_compact64_launch(const flash_mla::amd::fused_sparse::Args *p, int h, void *s) {
    return flash_mla::amd::compact_attention::launch_all<64>(p, h, s);
}
