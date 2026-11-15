// Independent native AMD GPU sparse attention. No external operator library.
// The original CUDA/Ascend paths and the unfused AMD baseline remain unchanged.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

#if defined(FLASH_MLA_AMD_REGISTER_Q) && defined(FLASH_MLA_AMD_QK_CHAINS)
#error "Select one QK operand strategy, not register-Q and accumulation chains together"
#endif

#ifndef FLASH_MLA_AMD_STAGE_MARK
#define FLASH_MLA_AMD_STAGE_MARK(index)                                                                                \
    do {                                                                                                               \
    } while (false)
#endif

namespace flash_mla::amd::fused_sparse {
using I4 = int32_t __attribute__((ext_vector_type(4)));
using U4 = uint32_t __attribute__((ext_vector_type(4)));
using F4 = float __attribute__((ext_vector_type(4)));
using B8 = __bf16 __attribute__((ext_vector_type(8)));
using F2 = float __attribute__((ext_vector_type(2)));
using B2 = __bf16 __attribute__((ext_vector_type(2)));

struct Args {
    const uint16_t *q;
    const uint8_t *cache, *extra;
    const int32_t *indices, *extra_indices, *lengths, *extra_lengths;
    const void *positions;
    const float *cos_sin, *sink;
    uint8_t *output;
    int32_t *scales;
    float *lse;
    unsigned long long *invocations;
    int64_t q_stride, cache_block_stride, cache_token_stride, extra_block_stride, extra_token_stride;
    int64_t index_stride, extra_index_stride, scale_pitch;
    int32_t rows, cache_blocks, page_size, topk, extra_blocks, extra_page_size, extra_topk;
    int32_t extra_format, cos_sin_stride, positions64, neox, already_roped;
    float softmax_scale;
};

__device__ __forceinline__ float bf(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ __forceinline__ uint32_t pack2(float a, float b) {
    return __builtin_bit_cast(uint32_t, __builtin_convertvector((F2){a, b}, B2));
}
__device__ __forceinline__ float bround(float x) { return bf(uint16_t(pack2(x, x))); }
__device__ __forceinline__ float ex(float x) { return __builtin_amdgcn_exp2f(x * 1.4426950408889634f); }
__device__ __forceinline__ F4 mma(I4 a, I4 b, F4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(B8, a), __builtin_bit_cast(B8, b), c, 0, 0, 0);
}
__device__ __forceinline__ int sw(int row, int col) { return row * 512 + (((col >> 3) ^ (row & 7)) << 3) + (col & 7); }
__device__ __forceinline__ int psw(int row, int col) { return row * 32 + (((col >> 3) ^ (row & 3)) << 3) + (col & 7); }
#ifdef FLASH_MLA_AMD_NATIVE_TRANSPOSE
using I2 = int32_t __attribute__((ext_vector_type(2)));
using S4 = int16_t __attribute__((ext_vector_type(4)));
__device__ __forceinline__ int native_physical(int row, int col) {
    const int key = ((row & 3) << 4) ^ (((row >> 3) & 1) << 6);
    return row * 512 + (col ^ key);
}
__device__ __forceinline__ int kv_sw(int row, int col) {
    const int physical = (row & 3) | (((row >> 3) & 3) << 2) | (((row >> 2) & 1) << 4);
    return native_physical(physical, col);
}
__device__ __forceinline__ I4 transposed_v(const uint16_t *kv, int col, int lane) {
    // The read returns four physical rows per 16-lane group. The staging row
    // permutation makes two reads concatenate into eight logical K rows.
    using LDS = __attribute__((address_space(3))) S4 *;
    const int row = lane >> 2, c = col + (lane & 3) * 4;
    const auto lo =
        __builtin_bit_cast(I2, __builtin_amdgcn_ds_read_tr16_b64_v4i16((LDS)(kv + native_physical(row, c))));
    const auto hi =
        __builtin_bit_cast(I2, __builtin_amdgcn_ds_read_tr16_b64_v4i16((LDS)(kv + native_physical(row + 16, c))));
    return I4{lo[0], lo[1], hi[0], hi[1]};
}
#else
__device__ __forceinline__ int kv_sw(int row, int col) { return sw(row, col); }
#endif
__device__ __forceinline__ float e4(uint8_t x) {
    // Native OCP E4M3 conversion, not a host-side unpacked cache.
    const auto v = __builtin_amdgcn_cvt_pk_f32_fp8(uint32_t(x), false);
    return v[0];
}
__device__ __forceinline__ float ue8(uint8_t x) {
    return __uint_as_float(x == 0 ? 0x00400000u : x == 255 ? 0x7fc00000u : uint32_t(x) << 23);
}
template <int BYTE> __device__ __forceinline__ uint32_t fp4_pair(uint32_t data, float scale) {
    // The native scaled conversion is an exponent-scale operation. The cache
    // uses E4M3 scales, whose mantissa must not be discarded as for UE8M0.
    const B2 raw = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(data, 1.f, BYTE);
    const F2 value = __builtin_convertvector(raw, F2);
    return pack2(value[0] * scale, value[1] * scale);
}
template <bool NEOX>
__device__ __forceinline__ void rope(float (&v)[8], const float *cs, int col, int lane, bool inverse) {
    if constexpr (NEOX) {
        float partner[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            partner[i] = __shfl_xor(v[i], 4, 64);
        if (lane >= 56) {
            const bool first = lane < 60;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int pair = ((col - 448) & 31) + i;
                const float c = cs[pair], s = cs[pair + 32], a = first ? v[i] : partner[i],
                            b = first ? partner[i] : v[i];
                v[i] = inverse ? (first ? fmaf(a, c, b * s) : fmaf(a, -s, b * c))
                               : (first ? fmaf(a, c, -(b * s)) : fmaf(a, s, b * c));
            }
        }
    } else if (lane >= 56) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int pair = (col - 448 + i) / 2;
            const float c = cs[pair], s = cs[pair + 32], a = v[i], b = v[i + 1];
            if (inverse) {
                v[i] = fmaf(a, c, b * s);
                v[i + 1] = fmaf(a, -s, b * c);
            } else {
                v[i] = fmaf(a, c, -(b * s));
                v[i + 1] = fmaf(a, s, b * c);
            }
        }
    }
}
template <int HEADS> struct Layout {
#ifdef FLASH_MLA_AMD_WIDE_WAVES
    static constexpr int NPARTS = 4, W = HEADS / 4, THREADS = W * 64, OUT = 8;
#else
    static constexpr int NPARTS = 2, W = HEADS / 8, THREADS = W * 64, OUT = 16;
#endif
    static constexpr int Q = 0;
#ifdef FLASH_MLA_AMD_REGISTER_Q
    static constexpr int KV = 0, VT = KV + 32 * 512 * 2;
#else
    static constexpr int KV = Q + HEADS * 512 * 2, VT = KV + 32 * 512 * 2;
#endif
#ifdef FLASH_MLA_AMD_NATIVE_TRANSPOSE
    static constexpr int SCORE = VT, PROB = SCORE + HEADS * 36 * 4;
#else
    static constexpr int SCORE = VT + 512 * 40 * 2, PROB = SCORE + HEADS * 36 * 4;
#endif
    static constexpr int STATE = PROB + HEADS * 32 * 2, VALID = STATE + HEADS * 4 * 4;
    static constexpr int WORK = (VALID + 32 * 4 + 255) / 256 * 256;
#ifdef FLASH_MLA_AMD_REGISTER_Q
    static constexpr int BYTES = WORK > HEADS * 512 * 2 ? WORK : HEADS * 512 * 2;
#else
    static constexpr int BYTES = WORK;
#endif
};

template <int HEADS, bool NEOX, bool POS64> __global__ __launch_bounds__(Layout<HEADS>::THREADS) void kernel(Args p) {
    constexpr int W = Layout<HEADS>::W, T = Layout<HEADS>::THREADS;
    extern __shared__ __attribute__((aligned(16))) uint8_t shared[];
    auto *q = reinterpret_cast<uint16_t *>(shared + Layout<HEADS>::Q);
    auto *kv = reinterpret_cast<uint16_t *>(shared + Layout<HEADS>::KV);
    auto *vt = reinterpret_cast<uint16_t *>(shared + Layout<HEADS>::VT);
    auto *score = reinterpret_cast<float *>(shared + Layout<HEADS>::SCORE);
    auto *prob = reinterpret_cast<uint16_t *>(shared + Layout<HEADS>::PROB);
    auto *state = reinterpret_cast<float *>(shared + Layout<HEADS>::STATE);
    auto *valid = reinterpret_cast<int *>(shared + Layout<HEADS>::VALID);
    const int tid = threadIdx.x, lane = tid % 64, r = lane % 16, g = lane / 16;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int row = blockIdx.x, head_base = blockIdx.y * HEADS;
    FLASH_MLA_AMD_STAGE_MARK(0);
    const int64_t pos =
        POS64 ? static_cast<const int64_t *>(p.positions)[row] : static_cast<const int32_t *>(p.positions)[row];
    const float *cs = p.cos_sin + pos * p.cos_sin_stride;
    if (p.invocations && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        atomicAdd(p.invocations, 1ULL);
#ifdef FLASH_MLA_AMD_REGISTER_Q
    static_assert(Layout<HEADS>::NPARTS == 2, "register-Q uses the two-part PV geometry");
    I4 query[16];
    const int query_head = (wave / 2) * 16 + r;
#pragma unroll
    for (int k = 0; k < 16; ++k)
        query[k] = *reinterpret_cast<const I4 *>(p.q + int64_t(row) * p.q_stride + (head_base + query_head) * 512 +
                                                 k * 32 + g * 8);
    if (!p.already_roped) {
        const U4 raw0 = __builtin_bit_cast(U4, query[14]), raw1 = __builtin_bit_cast(U4, query[15]);
        float a[8], b[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i * 2] = bf(raw0[i]);
            a[i * 2 + 1] = bf(raw0[i] >> 16);
            b[i * 2] = bf(raw1[i]);
            b[i * 2 + 1] = bf(raw1[i] >> 16);
        }
        if constexpr (NEOX) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float c = cs[g * 8 + i], s = cs[32 + g * 8 + i], x = a[i], y = b[i];
                a[i] = fmaf(x, c, -y * s);
                b[i] = fmaf(x, s, y * c);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                float c = cs[g * 4 + i / 2], s = cs[32 + g * 4 + i / 2], x = a[i], y = a[i + 1];
                a[i] = fmaf(x, c, -y * s);
                a[i + 1] = fmaf(x, s, y * c);
                c = cs[16 + g * 4 + i / 2];
                s = cs[48 + g * 4 + i / 2];
                x = b[i];
                y = b[i + 1];
                b[i] = fmaf(x, c, -y * s);
                b[i + 1] = fmaf(x, s, y * c);
            }
        }
        I4 lo, hi;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lo[i] = pack2(a[i * 2], a[i * 2 + 1]);
            hi[i] = pack2(b[i * 2], b[i * 2 + 1]);
        }
        query[14] = lo;
        query[15] = hi;
    }
#else
    for (int h = wave; h < HEADS; h += W) {
        const int col = lane * 8;
        const U4 raw = *reinterpret_cast<const U4 *>(p.q + int64_t(row) * p.q_stride + (head_base + h) * 512 + col);
        float v[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i] = bf(uint16_t(raw[i]));
            v[2 * i + 1] = bf(uint16_t(raw[i] >> 16));
        }
        if (!p.already_roped)
            rope<NEOX>(v, cs, col, lane, false);
        U4 enc;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            enc[i] = pack2(v[i * 2], v[i * 2 + 1]);
        *reinterpret_cast<U4 *>(q + sw(h, col)) = enc;
    }
#endif
    if (tid < HEADS) {
        state[tid * 4] = -INFINITY;
        state[tid * 4 + 1] = 0.f;
        state[tid * 4 + 2] = 0.f;
        state[tid * 4 + 3] = 0.f;
    }
    __syncthreads();
    FLASH_MLA_AMD_STAGE_MARK(1);
    F4 output[Layout<HEADS>::OUT] = {};
    const int main_length = p.lengths ? min(max(p.lengths[row], 0), p.topk) : p.topk;
    const int extra_length = p.extra_lengths ? min(max(p.extra_lengths[row], 0), p.extra_topk) : p.extra_topk;
    const int main_tiles = (main_length + 31) / 32, extra_tiles = (extra_length + 31) / 32;
    for (int tile = 0; tile < main_tiles + extra_tiles; ++tile) {
        const bool extra = tile >= main_tiles;
        const int begin = (extra ? tile - main_tiles : tile) * 32;
        const int length = extra ? extra_length : main_length;
        const int32_t *index =
            (extra ? p.extra_indices : p.indices) + int64_t(row) * (extra ? p.extra_index_stride : p.index_stride);
        const uint8_t *cache = extra ? p.extra : p.cache;
        const int page = extra ? p.extra_page_size : p.page_size, blocks = extra ? p.extra_blocks : p.cache_blocks;
        const int64_t bs = extra ? p.extra_block_stride : p.cache_block_stride,
                      ts = extra ? p.extra_token_stride : p.cache_token_stride;
        const bool fp4 = extra && p.extra_format == 4;
        // Each thread gathers eight adjacent dequantized values. A vector of
        // FP4 values uses four packed native conversions, not scalar LUT work.
#ifdef FLASH_MLA_AMD_BUFFER_GATHER
#pragma unroll 2
#elif defined(FLASH_MLA_AMD_NATIVE_TRANSPOSE)
#pragma clang loop unroll(disable)
#endif
        for (int batch = 0; batch < 32 / W; ++batch) {
            // A wave owns one selected row. Make this uniformity explicit so
            // cache addressing and page division do not become per-lane work.
            const int kr = wave + batch * W, col = lane * 8;
            const int token = __builtin_amdgcn_readfirstlane(begin + kr < length ? index[begin + kr] : -1);
            const bool ok = token >= 0 && int64_t(token) < int64_t(blocks) * page;
            if (col == 0)
                valid[kr] = int(ok);
            U4 bits{};
#ifdef FLASH_MLA_AMD_BUFFER_GATHER
            {
                // A descriptor covers one row, never the entire cache. The
                // selected row base is computed in int64 before constructing
                // it, so caches larger than 4GiB retain correct addressing.
                const int selected = ok ? token : 0;
#else
            if (ok) {
                const int selected = token;
#endif
                const int block = page == 128 ? selected >> 7 : selected / page;
                const int offset = page == 128 ? selected & 127 : selected % page;
                const uint8_t *src = cache + int64_t(block) * bs + int64_t(offset) * ts;
#ifdef FLASH_MLA_AMD_BUFFER_GATHER
                const auto resource = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(src), short(0),
                                                                        ok ? (fp4 ? 288 : 528) : 0, 0x00020000);
#endif
                if (fp4) {
#ifdef FLASH_MLA_AMD_BUFFER_GATHER
                    const uint32_t packed = __builtin_amdgcn_raw_buffer_load_b32(resource, col / 2, 0, 0);
                    const uint32_t scales =
                        __builtin_amdgcn_raw_buffer_load_b32(resource, 256 + ((col / 16) & ~3), 0, 0);
                    const float scale = e4(uint8_t(scales >> ((col / 16 % 4) * 8)));
#else
                    const uint32_t packed = *reinterpret_cast<const uint32_t *>(src + col / 2);
                    const float scale = e4(src[256 + col / 16]);
#endif
                    bits[0] = fp4_pair<0>(packed, scale);
                    bits[1] = fp4_pair<1>(packed, scale);
                    bits[2] = fp4_pair<2>(packed, scale);
                    bits[3] = fp4_pair<3>(packed, scale);
                } else {
#ifdef FLASH_MLA_AMD_BUFFER_GATHER
                    const uint32_t lo = __builtin_amdgcn_raw_buffer_load_b32(resource, col, 0, 0);
                    const uint32_t hi = __builtin_amdgcn_raw_buffer_load_b32(resource, col + 4, 0, 0);
                    const uint64_t packed = uint64_t(lo) | (uint64_t(hi) << 32);
                    const uint32_t scales =
                        __builtin_amdgcn_raw_buffer_load_b32(resource, 512 + ((col / 32) & ~3), 0, 0);
                    const float scale = ue8(uint8_t(scales >> ((col / 32 % 4) * 8)));
#else
                    const uint64_t packed = *reinterpret_cast<const uint64_t *>(src + col);
                    const float scale = ue8(src[512 + col / 32]);
#endif
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const auto x = __builtin_amdgcn_cvt_pk_f32_fp8(uint32_t(packed >> (j * 16)), false);
                        bits[j] = pack2(x[0] * scale, x[1] * scale);
                    }
                }
            }
            *reinterpret_cast<U4 *>(kv + kv_sw(kr, col)) = bits;
        }
        __syncthreads();
        FLASH_MLA_AMD_STAGE_MARK(2 + tile * 4);
#ifndef FLASH_MLA_AMD_NATIVE_TRANSPOSE
        // Scalar transposed stores make a wave's adjacent eight-value chunks
        // hit the same LDS bank. Instead read the staged rows coalesced and
        // write eight contiguous keys as one vector. The later score barrier
        // also completes this transpose before any PV consumer reads it.
        for (int i = tid; i < 512 * 4; i += T) {
            const int d = i % 512, k = (i / 512) * 8;
            U4 value;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                value[j] = uint32_t(kv[sw(k + j * 2, d)]) | (uint32_t(kv[sw(k + j * 2 + 1, d)]) << 16);
            *reinterpret_cast<U4 *>(vt + d * 40 + k) = value;
        }
#endif
        // D += Q K^T. A swapped operand order gives head=lane%16 and
        // key=4*(lane/16)+element in the native BF16 MFMA result.
        if (wave < HEADS / 8) {
            const int qh = (wave / 2) * 16 + r, kk = (wave % 2) * 16 + r;
            F4 s = {};
#ifdef FLASH_MLA_AMD_QK_CHAINS
            F4 chains[4] = {};
#pragma unroll
            for (int k = 0; k < 512; k += 128) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const I4 qa = *reinterpret_cast<const I4 *>(q + sw(qh, k + j * 32 + g * 8));
                    const I4 ka = *reinterpret_cast<const I4 *>(kv + kv_sw(kk, k + j * 32 + g * 8));
                    chains[j] = mma(ka, qa, chains[j]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
            // Same BF16 operands and FP32 accumulation, but a different FP32
            // reduction order. This variant requires its own numerical gate.
            s = ((chains[0] + chains[1]) + chains[2]) + chains[3];
#else
#ifdef FLASH_MLA_AMD_REGISTER_Q
#pragma unroll
#else
#pragma unroll 4
#endif
            for (int k = 0; k < 512; k += 32) {
#ifdef FLASH_MLA_AMD_REGISTER_Q
                const I4 qa = query[k / 32];
#else
                const I4 qa = *reinterpret_cast<const I4 *>(q + sw(qh, k + g * 8));
#endif
                const I4 ka = *reinterpret_cast<const I4 *>(kv + kv_sw(kk, k + g * 8));
                s = mma(ka, qa, s);
#ifdef FLASH_MLA_AMD_REGISTER_Q
                __builtin_amdgcn_sched_barrier(0);
#endif
            }
#endif
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = (wave % 2) * 16 + g * 4 + j;
                score[qh * 36 + key] = valid[key] ? s[j] * p.softmax_scale : -INFINITY;
            }
        }
        __syncthreads();
        FLASH_MLA_AMD_STAGE_MARK(3 + tile * 4);
        // Two adjacent keys per lane, 16-lane row groups. Intermediate P
        // remains BF16, matching the existing online-softmax/PV contract.
        for (int h = tid / 16; h < HEADS; h += T / 16) {
            const int key = (tid % 16) * 2;
            const float s0 = score[h * 36 + key], s1 = score[h * 36 + key + 1];
            float maximum = fmaxf(s0, s1);
#pragma unroll
            for (int mask = 8; mask; mask >>= 1)
                maximum = fmaxf(maximum, __shfl_xor(maximum, mask, 16));
            const float next = fmaxf(state[h * 4], maximum), safe = next == -INFINITY ? 0.f : next;
            const float alpha = ex(state[h * 4] - safe), v0 = ex(s0 - safe), v1 = ex(s1 - safe);
            float sum = v0 + v1;
#pragma unroll
            for (int mask = 8; mask; mask >>= 1)
                sum += __shfl_xor(sum, mask, 16);
            *reinterpret_cast<uint32_t *>(prob + psw(h, key)) = pack2(v0, v1);
            if (tid % 16 == 0) {
                state[h * 4] = next;
                state[h * 4 + 1] = state[h * 4 + 1] * alpha + sum;
                state[h * 4 + 2] = alpha;
            }
        }
        __syncthreads();
        FLASH_MLA_AMD_STAGE_MARK(4 + tile * 4);
        const int ph = (wave / Layout<HEADS>::NPARTS) * 16 + r,
                  n0 = (wave % Layout<HEADS>::NPARTS) * (512 / Layout<HEADS>::NPARTS);
        const I4 a = *reinterpret_cast<const I4 *>(prob + psw(ph, g * 8));
        const float alpha = state[ph * 4 + 2];
#pragma unroll
        for (int j = 0; j < Layout<HEADS>::OUT; ++j) {
            output[j] *= alpha;
#ifdef FLASH_MLA_AMD_NATIVE_TRANSPOSE
            const I4 b = transposed_v(kv, n0 + j * 16, lane);
#else
            const I4 b = *reinterpret_cast<const I4 *>(vt + (n0 + j * 16 + r) * 40 + g * 8);
#endif
            output[j] = mma(b, a, output[j]);
        }
        __syncthreads();
        FLASH_MLA_AMD_STAGE_MARK(5 + tile * 4);
    }
    // Normalize once. Preserve the old BF16 narrowing BEFORE inverse RoPE.
    const int ph = (wave / Layout<HEADS>::NPARTS) * 16 + r,
              n0 = (wave % Layout<HEADS>::NPARTS) * (512 / Layout<HEADS>::NPARTS);
    const float den = state[ph * 4 + 1], inv = den > 0 ? 1.f / den : 1.f;
    const float logsum = den > 0 ? state[ph * 4] + logf(den) : INFINITY;
    const float sink = den > 0 ? (p.sink ? 1.f / (1.f + ex(p.sink[head_base + ph] - logsum)) : 1.f) : 0.f;
    if (p.lse && g == 0 && wave % Layout<HEADS>::NPARTS == 0)
        p.lse[int64_t(row) * 64 + head_base + ph] = logsum;
#ifdef FLASH_MLA_AMD_REGISTER_Q
    // The final BF16 output reuses the completed KV/score workspace. Every
    // wave must first read its normalization state before this overwrite.
    __syncthreads();
#endif
#pragma unroll
    for (int j = 0; j < Layout<HEADS>::OUT; ++j) {
        const int d = n0 + j * 16 + g * 4;
        const uint32_t a = pack2((output[j][0] * inv) * sink, (output[j][1] * inv) * sink);
        const uint32_t b = pack2((output[j][2] * inv) * sink, (output[j][3] * inv) * sink);
        *reinterpret_cast<uint64_t *>(q + sw(ph, d)) = uint64_t(a) | (uint64_t(b) << 32);
    }
    __syncthreads();
    for (int h = wave; h < HEADS; h += W) {
        const int col = lane * 8;
        const U4 raw = *reinterpret_cast<const U4 *>(q + sw(h, col));
        float v[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i] = bf(uint16_t(raw[i]));
            v[2 * i + 1] = bf(uint16_t(raw[i] >> 16));
        }
        rope<NEOX>(v, cs, col, lane, true);
        float maximum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = bround(v[i]);
            maximum = fmaxf(maximum, fabsf(v[i]));
        }
        maximum = fmaxf(maximum, __shfl_xor(maximum, 1, 64));
        maximum = fmaxf(maximum, __shfl_xor(maximum, 2, 64));
        const uint32_t scale_bits = __float_as_uint(fmaxf(maximum, 1e-4f) * (1.f / 448.f));
        const int exponent = int((scale_bits - 1) >> 23) + 1;
        const float inverse = __uint_as_float(uint32_t(254 - exponent) << 23);
        U4 narrow{};
        int ab = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] * inverse, v[1] * inverse, 0, false);
        narrow[0] = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] * inverse, v[3] * inverse, ab, true);
        ab = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] * inverse, v[5] * inverse, 0, false);
        narrow[1] = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] * inverse, v[7] * inverse, ab, true);
        *reinterpret_cast<uint64_t *>(p.output + int64_t(row) * 32768 + (head_base + h) * 512 + col) =
            uint64_t(narrow[0]) | (uint64_t(narrow[1]) << 32);
        uint32_t sf = exponent;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 4, 64)) << 8;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 8, 64)) << 16;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 12, 64)) << 24;
        if (lane % 16 == 0)
            p.scales[row + int64_t((head_base + h) * 4 + lane / 16) * p.scale_pitch] = int32_t(sf);
    }
    FLASH_MLA_AMD_STAGE_MARK(2 + (main_tiles + extra_tiles) * 4);
}

template <int H, bool N, bool I> int prepare() {
    return hipFuncSetAttribute((const void *)kernel<H, N, I>, hipFuncAttributeMaxDynamicSharedMemorySize,
                               Layout<H>::BYTES);
}
template <int H, bool N, bool I> int launch(Args p, hipStream_t s) {
    hipLaunchKernelGGL((kernel<H, N, I>), dim3(p.rows, 64 / H), dim3(Layout<H>::THREADS), Layout<H>::BYTES, s, p);
    return hipGetLastError();
}
} // namespace flash_mla::amd::fused_sparse

extern "C" size_t fused_attention_args_size() { return sizeof(flash_mla::amd::fused_sparse::Args); }
extern "C" int fused_attention_prepare(int heads) {
    using namespace flash_mla::amd::fused_sparse;
    int status = 0;
#define PREP(H)                                                                                                        \
    status = prepare<H, false, false>();                                                                               \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, false, true>();                                                                                \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    status = prepare<H, true, false>();                                                                                \
    if (status)                                                                                                        \
        return status;                                                                                                 \
    return prepare<H, true, true>();
    if (heads == 32) {
        PREP(32)
    }
    if (heads == 64) {
        PREP(64)
    }
#undef PREP
    return hipErrorInvalidValue;
}
extern "C" int fused_attention_launch(const flash_mla::amd::fused_sparse::Args *p, int heads, void *stream) {
    using namespace flash_mla::amd::fused_sparse;
    if (!p || !p->q || !p->cache || !p->indices || !p->positions || !p->cos_sin || !p->output || !p->scales ||
        p->rows < 1 || p->rows > 1536 || p->q_stride < 32768 || p->page_size < 1 || p->cache_blocks < 1 ||
        p->topk < 0 || p->topk > 640 || p->extra_topk < 0 || p->extra_topk > 512 || p->index_stride < p->topk ||
        p->scale_pitch < p->rows || p->cos_sin_stride < 64 || (p->positions64 != 0 && p->positions64 != 1) ||
        (p->neox != 0 && p->neox != 1) || (p->already_roped != 0 && p->already_roped != 1) ||
        !std::isfinite(p->softmax_scale) || p->softmax_scale <= 0)
        return hipErrorInvalidValue;
    if (p->extra_topk && (!p->extra || !p->extra_indices || p->extra_blocks < 1 || p->extra_page_size < 1 ||
                          p->extra_index_stride < p->extra_topk || (p->extra_format != 4 && p->extra_format != 8)))
        return hipErrorInvalidValue;
    auto s = static_cast<hipStream_t>(stream);
#define RUN(H)                                                                                                         \
    if (p->neox)                                                                                                       \
        return p->positions64 ? launch<H, true, true>(*p, s) : launch<H, true, false>(*p, s);                          \
    return p->positions64 ? launch<H, false, true>(*p, s) : launch<H, false, false>(*p, s);
    if (heads == 32) {
        RUN(32)
    }
    if (heads == 64) {
        RUN(64)
    }
#undef RUN
    return hipErrorInvalidValue;
}
