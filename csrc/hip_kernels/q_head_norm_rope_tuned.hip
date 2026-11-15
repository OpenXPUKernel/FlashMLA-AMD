// Independent AMD GPU tuning entries; the original transform is unchanged.
#include "q_head_norm_rope.hip"

namespace flash_mla::amd::q_head_tuned {
using namespace flash_mla::amd::post_projection;
using BaseArgs = flash_mla::amd::q_head_transform::Args;
using U4 = uint32_t __attribute__((ext_vector_type(4)));
using F8 = float __attribute__((ext_vector_type(8)));
using B8 = __bf16 __attribute__((ext_vector_type(8)));
struct Args {
    BaseArgs base;
    unsigned long long *invocations;
};

template <int WAVES, int GROUP, int POLICY, bool PACK_ONCE, bool NEOX, bool POS64>
__global__ __launch_bounds__(WAVES * 64) void transform(Args full) {
    const auto p = full.base;
    const int lane = threadIdx.x % 64;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int base = (blockIdx.x * WAVES + wave) * GROUP;
    const int col = lane * 8;
    const U4 weights = *reinterpret_cast<const U4 *>(p.weight + col);
    if (full.invocations && blockIdx.x == 0 && threadIdx.x == 0)
        atomicAdd(full.invocations, 1ULL);
#pragma unroll
    for (int item = 0; item < GROUP; ++item) {
        const int row = base + item;
        if (row >= p.tokens * 64)
            break;
        const int token = row >> 6, head = row & 63;
        const auto *src =
            reinterpret_cast<const U4 *>(p.input + int64_t(token) * p.input_token_stride + head * 512 + col);
        U4 raw;
        if constexpr (POLICY & 2)
            raw = __builtin_nontemporal_load(src);
        else
            raw = *src;
        float v[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i] = bf16(uint16_t(raw[i]));
            v[2 * i + 1] = bf16(uint16_t(raw[i] >> 16));
        }
        const float inv = rsqrtf(baseline_kv_sum(v) * (1.f / 512.f) + p.epsilon);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i] = (v[2 * i] * inv) * bf16(uint16_t(weights[i]));
            v[2 * i + 1] = (v[2 * i + 1] * inv) * bf16(uint16_t(weights[i] >> 16));
        }
        U4 encoded;
        if constexpr (PACK_ONCE) {
            const F8 values = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
            encoded = __builtin_bit_cast(U4, __builtin_convertvector(values, B8));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[2 * i] = bf16(uint16_t(encoded[i]));
                v[2 * i + 1] = bf16(uint16_t(encoded[i] >> 16));
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; i += 2)
                rounded_pair(v[i], v[i + 1]);
        }
        const int64_t pos =
            POS64 ? static_cast<const int64_t *>(p.positions)[token] : static_cast<const int32_t *>(p.positions)[token];
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
                    const float c = p.cos_sin[pos * p.cos_sin_stride + pair],
                                s = p.cos_sin[pos * p.cos_sin_stride + pair + 32];
                    const float a = first ? v[i] : partner[i], b = first ? partner[i] : v[i];
                    v[i] = first ? fmaf(a, c, -(b * s)) : fmaf(a, s, b * c);
                }
            }
        } else if (lane >= 56) {
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int pair = (col - 448 + i) / 2;
                const float c = p.cos_sin[pos * p.cos_sin_stride + pair],
                            s = p.cos_sin[pos * p.cos_sin_stride + pair + 32];
                const float a = v[i], b = v[i + 1];
                v[i] = fmaf(a, c, -(b * s));
                v[i + 1] = fmaf(a, s, b * c);
            }
        }
        if constexpr (PACK_ONCE) {
            if (lane >= 56) {
                const F8 values = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
                encoded = __builtin_bit_cast(U4, __builtin_convertvector(values, B8));
            }
        } else {
            const F8 values = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
            encoded = __builtin_bit_cast(U4, __builtin_convertvector(values, B8));
        }
        auto *dst = reinterpret_cast<U4 *>(p.output + int64_t(row) * 512 + col);
        if constexpr (POLICY & 1)
            __builtin_nontemporal_store(encoded, dst);
        else
            *dst = encoded;
    }
}
template <int W, int G, int P, bool B, bool N, bool I> int launch(Args p, hipStream_t stream) {
    const int rows = p.base.tokens * 64;
    hipLaunchKernelGGL((transform<W, G, P, B, N, I>), dim3((rows + W * G - 1) / (W * G)), dim3(W * 64), 0, stream, p);
    return hipGetLastError();
}
template <int W, int G, int P, bool B> int dispatch(Args p, hipStream_t stream) {
    if (p.base.neox)
        return p.base.positions64 ? launch<W, G, P, B, true, true>(p, stream)
                                  : launch<W, G, P, B, true, false>(p, stream);
    return p.base.positions64 ? launch<W, G, P, B, false, true>(p, stream)
                              : launch<W, G, P, B, false, false>(p, stream);
}
} // namespace flash_mla::amd::q_head_tuned
extern "C" size_t q_head_tuned_args_size() { return sizeof(flash_mla::amd::q_head_tuned::Args); }
extern "C" int q_head_transform_tuned(const flash_mla::amd::q_head_tuned::Args *args, int mode, void *stream) {
    using namespace flash_mla::amd::q_head_tuned;
    if (!args)
        return hipErrorInvalidValue;
    const auto &p = args->base;
    if (!p.input || !p.weight || !p.cos_sin || !p.positions || !p.output || p.tokens < 1 || p.tokens > 1536 ||
        p.heads != 64 || p.input_token_stride < 32768 || p.input_token_stride % 8 || p.cos_sin_stride < 64 ||
        !std::isfinite(p.epsilon) || p.epsilon < 0 || (p.neox != 0 && p.neox != 1) ||
        (p.positions64 != 0 && p.positions64 != 1))
        return hipErrorInvalidValue;
    if ((reinterpret_cast<uintptr_t>(p.input) | reinterpret_cast<uintptr_t>(p.weight) |
         reinterpret_cast<uintptr_t>(p.output)) %
        16)
        return hipErrorInvalidValue;
    if (args->invocations && reinterpret_cast<uintptr_t>(args->invocations) % 8)
        return hipErrorInvalidValue;
    auto s = static_cast<hipStream_t>(stream);
    switch (mode) {
    case 0:
        return dispatch<1, 1, 0, false>(*args, s);
    case 1:
        return dispatch<1, 1, 1, false>(*args, s);
    case 2:
        return dispatch<1, 1, 3, false>(*args, s);
    case 3:
        return dispatch<1, 1, 0, true>(*args, s);
    case 4:
        return dispatch<1, 1, 1, true>(*args, s);
    case 5:
        return dispatch<1, 2, 1, true>(*args, s);
    case 6:
        return dispatch<1, 4, 1, true>(*args, s);
    case 7:
        return dispatch<2, 2, 1, true>(*args, s);
    case 8:
        return dispatch<4, 2, 1, true>(*args, s);
    case 9:
        return dispatch<1, 4, 0, true>(*args, s);
    case 10:
        return dispatch<1, 2, 0, true>(*args, s);
    case 11:
        return dispatch<1, 8, 1, true>(*args, s);
    }
    return hipErrorInvalidValue;
}
