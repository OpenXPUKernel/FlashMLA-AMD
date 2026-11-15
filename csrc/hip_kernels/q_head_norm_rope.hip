// Opt-in standalone AMD GPU Q-head transform. CUDA and other backends are
// unchanged. BF16 rounding and the existing reduction tree are preserved.
#include "post_projection.hip"

namespace flash_mla::amd::q_head_transform {
using namespace flash_mla::amd::post_projection;

struct Args {
    const uint16_t *input, *weight;
    const float *cos_sin;
    const void *positions;
    uint16_t *output;
    int64_t input_token_stride;
    int32_t tokens, heads, cos_sin_stride;
    float epsilon;
    int32_t neox, positions64;
};

template <int WAVES, bool NEOX, bool POS64> __global__ __launch_bounds__(64 * WAVES) void transform(Args p) {
    const int head_row = blockIdx.x * WAVES + threadIdx.x / 64;
    if (head_row >= p.tokens * p.heads)
        return;
    const int lane = threadIdx.x % 64;
    const int token = head_row / p.heads;
    const int head = head_row % p.heads;
    const int col = lane * 8;
    const uint4 raw =
        *reinterpret_cast<const uint4 *>(p.input + int64_t(token) * p.input_token_stride + head * 512 + col);
    const uint4 weights = *reinterpret_cast<const uint4 *>(p.weight + col);
    const uint32_t bits[4] = {raw.x, raw.y, raw.z, raw.w};
    const uint32_t wb[4] = {weights.x, weights.y, weights.z, weights.w};
    float v[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i * 2] = bf16(uint16_t(bits[i]));
        v[i * 2 + 1] = bf16(uint16_t(bits[i] >> 16));
    }
    const float inverse = rsqrtf(baseline_kv_sum(v) * (1.f / 512.f) + p.epsilon);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i * 2] = (v[i * 2] * inverse) * bf16(uint16_t(wb[i]));
        v[i * 2 + 1] = (v[i * 2 + 1] * inverse) * bf16(uint16_t(wb[i] >> 16));
        rounded_pair(v[i * 2], v[i * 2 + 1]);
    }
    const int64_t pos =
        POS64 ? static_cast<const int64_t *>(p.positions)[token] : static_cast<const int32_t *>(p.positions)[token];
    if constexpr (NEOX) {
        float other[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            other[i] = __shfl_xor(v[i], 4, 64);
        if (lane >= 56) {
            const bool first = lane < 60;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int pair = ((col - 448) & 31) + i;
                const float c = p.cos_sin[pos * p.cos_sin_stride + pair];
                const float s = p.cos_sin[pos * p.cos_sin_stride + pair + 32];
                const float a = first ? v[i] : other[i], b = first ? other[i] : v[i];
                v[i] = first ? fmaf(a, c, -(b * s)) : fmaf(a, s, b * c);
            }
        }
    } else if (lane >= 56) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int pair = (col - 448 + i) / 2;
            const float c = p.cos_sin[pos * p.cos_sin_stride + pair];
            const float s = p.cos_sin[pos * p.cos_sin_stride + pair + 32];
            const float a = v[i], b = v[i + 1];
            v[i] = fmaf(a, c, -(b * s));
            v[i + 1] = fmaf(a, s, b * c);
        }
    }
    typedef float v8f __attribute__((ext_vector_type(8)));
    typedef __bf16 v8b __attribute__((ext_vector_type(8)));
    const v8f result = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    *reinterpret_cast<uint4 *>(p.output + int64_t(head_row) * 512 + col) =
        __builtin_bit_cast(uint4, __builtin_convertvector(result, v8b));
}

template <int WAVES, bool NEOX, bool POS64> hipError_t launch(const Args &p, hipStream_t stream) {
    const int rows = p.tokens * p.heads;
    hipLaunchKernelGGL((transform<WAVES, NEOX, POS64>), dim3((rows + WAVES - 1) / WAVES), dim3(WAVES * 64), 0, stream,
                       p);
    return hipGetLastError();
}

template <int WAVES> hipError_t dispatch(const Args &p, hipStream_t s) {
    if (p.neox)
        return p.positions64 ? launch<WAVES, true, true>(p, s) : launch<WAVES, true, false>(p, s);
    return p.positions64 ? launch<WAVES, false, true>(p, s) : launch<WAVES, false, false>(p, s);
}
} // namespace flash_mla::amd::q_head_transform

extern "C" size_t q_head_args_size() { return sizeof(flash_mla::amd::q_head_transform::Args); }
extern "C" int q_head_transform(const flash_mla::amd::q_head_transform::Args *p, int waves, void *stream) {
    using namespace flash_mla::amd::q_head_transform;
    if (!p || !p->input || !p->weight || !p->cos_sin || !p->positions || !p->output || p->tokens < 1 ||
        p->tokens > 1536 || p->heads != 64 || p->input_token_stride < int64_t(p->heads) * 512 ||
        p->input_token_stride % 8 || p->cos_sin_stride < 64 || !std::isfinite(p->epsilon) || p->epsilon < 0 ||
        (p->neox != 0 && p->neox != 1) || (p->positions64 != 0 && p->positions64 != 1))
        return int(hipErrorInvalidValue);
    const uintptr_t alignment = reinterpret_cast<uintptr_t>(p->input) | reinterpret_cast<uintptr_t>(p->weight) |
                                reinterpret_cast<uintptr_t>(p->output);
    if (alignment % 16)
        return int(hipErrorInvalidValue);
    auto s = static_cast<hipStream_t>(stream);
    if (waves == 1)
        return int(dispatch<1>(*p, s));
    if (waves == 2)
        return int(dispatch<2>(*p, s));
    if (waves == 4)
        return int(dispatch<4>(*p, s));
    if (waves == 8)
        return int(dispatch<8>(*p, s));
    return int(hipErrorInvalidValue);
}
