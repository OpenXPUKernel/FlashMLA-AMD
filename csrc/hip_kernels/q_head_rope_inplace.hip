// AMD-only opt-in for an exclusively owned Q projection with Q norm disabled.
// Preserve the other head dimensions and any adjacent index-query projection.
#include "q_head_norm_rope_tuned.hip"

namespace flash_mla::amd::q_rope {
using namespace flash_mla::amd::post_projection;
struct Args {
    uint16_t *query;
    const float *cos_sin;
    const void *positions;
    unsigned long long *invocations;
    int64_t token_stride;
    int32_t tokens, cos_sin_stride, neox, positions64;
};
template <int WAVES, bool NEOX, bool POS64> __global__ __launch_bounds__(WAVES * 64) void rotate(Args p) {
    const int row = blockIdx.x * (WAVES * 4) + threadIdx.x / 16;
    if (row >= p.tokens * 64)
        return;
    const int token = __builtin_amdgcn_readfirstlane(row >> 6), head = row & 63;
    const int lane = threadIdx.x % 16, col = lane * 4;
    auto *address = reinterpret_cast<uint2 *>(p.query + int64_t(token) * p.token_stride + head * 512 + 448 + col);
    const uint2 raw = *address;
    float v[4] = {bf16(uint16_t(raw.x)), bf16(uint16_t(raw.x >> 16)), bf16(uint16_t(raw.y)),
                  bf16(uint16_t(raw.y >> 16))};
    const int64_t pos =
        POS64 ? static_cast<const int64_t *>(p.positions)[token] : static_cast<const int32_t *>(p.positions)[token];
    if constexpr (NEOX) {
        float other[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            other[i] = __shfl_xor(v[i], 8, 16);
        const bool first = lane < 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int pair = (col & 31) + i;
            const float c = p.cos_sin[pos * p.cos_sin_stride + pair], s = p.cos_sin[pos * p.cos_sin_stride + pair + 32];
            const float a = first ? v[i] : other[i], b = first ? other[i] : v[i];
            v[i] = first ? fmaf(a, c, -(b * s)) : fmaf(a, s, b * c);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
            const int pair = (col + i) / 2;
            const float c = p.cos_sin[pos * p.cos_sin_stride + pair], s = p.cos_sin[pos * p.cos_sin_stride + pair + 32];
            const float a = v[i], b = v[i + 1];
            v[i] = fmaf(a, c, -(b * s));
            v[i + 1] = fmaf(a, s, b * c);
        }
    }
    using F4 = float __attribute__((ext_vector_type(4)));
    using B4 = __bf16 __attribute__((ext_vector_type(4)));
    *address = __builtin_bit_cast(uint2, __builtin_convertvector((F4){v[0], v[1], v[2], v[3]}, B4));
    if (p.invocations && blockIdx.x == 0 && threadIdx.x == 0)
        atomicAdd(p.invocations, 1ULL);
}
template <int W, bool N, bool I> int launch(Args p, hipStream_t s) {
    const int rows = p.tokens * 64;
    hipLaunchKernelGGL((rotate<W, N, I>), dim3((rows + W * 4 - 1) / (W * 4)), dim3(W * 64), 0, s, p);
    return hipGetLastError();
}
template <int W> int dispatch(Args p, hipStream_t s) {
    if (p.neox)
        return p.positions64 ? launch<W, true, true>(p, s) : launch<W, true, false>(p, s);
    return p.positions64 ? launch<W, false, true>(p, s) : launch<W, false, false>(p, s);
}
} // namespace flash_mla::amd::q_rope
extern "C" size_t q_rope_args_size() { return sizeof(flash_mla::amd::q_rope::Args); }
extern "C" int q_head_rope_inplace(const flash_mla::amd::q_rope::Args *p, int waves, void *stream) {
    using namespace flash_mla::amd::q_rope;
    if (!p || !p->query || !p->cos_sin || !p->positions || p->tokens < 1 || p->tokens > 1536 ||
        p->token_stride < 32768 || p->token_stride >= (1LL << 31) || p->token_stride % 8 || p->cos_sin_stride < 64 ||
        (p->neox != 0 && p->neox != 1) || (p->positions64 != 0 && p->positions64 != 1))
        return hipErrorInvalidValue;
    if (reinterpret_cast<uintptr_t>(p->query) % 16 ||
        (p->invocations && reinterpret_cast<uintptr_t>(p->invocations) % 8))
        return hipErrorInvalidValue;
    auto s = static_cast<hipStream_t>(stream);
    if (waves == 1)
        return dispatch<1>(*p, s);
    if (waves == 2)
        return dispatch<2>(*p, s);
    if (waves == 4)
        return dispatch<4>(*p, s);
    if (waves == 8)
        return dispatch<8>(*p, s);
    return hipErrorInvalidValue;
}
