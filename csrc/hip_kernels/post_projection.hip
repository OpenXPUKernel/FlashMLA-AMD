// Native AMD GPU post-projection transforms. Standalone C ABI for validation;
// no change to CUDA/Ascend dispatch and no external operator-library dependency.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

namespace flash_mla::amd::post_projection {

struct Args {
    const uint16_t *q, *kv, *q_weight, *kv_weight;
    const float *cos_sin;
    const void *positions;
    const int32_t *slots;
    uint8_t *q_out;
    int32_t *q_sf;
    uint8_t *cache;
    int64_t q_stride, kv_stride, sf_pitch, cache_block_stride, cache_token_stride;
    int32_t rows, cache_blocks, page_size, cos_sin_stride;
    float q_eps, kv_eps, q_scale;
    int32_t neox, positions64;
};

__device__ __forceinline__ float bf16(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }
__device__ __forceinline__ float rounded_bf16(float f) {
    uint32_t bits = __float_as_uint(f);
    bits += 0x7fff + ((bits >> 16) & 1);
    return __uint_as_float(bits & 0xffff0000u);
}
__device__ __forceinline__ void rounded_pair(float &a, float &b) {
    typedef float v2f __attribute__((ext_vector_type(2)));
    typedef __bf16 v2bf __attribute__((ext_vector_type(2)));
    const uint32_t packed = __builtin_bit_cast(uint32_t, __builtin_convertvector((v2f){a, b}, v2bf));
    a = __uint_as_float(packed << 16);
    b = __uint_as_float(packed & 0xffff0000u);
}
__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int mask = 32; mask; mask >>= 1)
        x += __shfl_xor(x, mask, 64);
    return x;
}
__device__ __forceinline__ float baseline_wave_sum(float x) {
    // Match the baseline's DPP row_shr 8/4/2/1, row_bcast15, row_bcast31
    // reduction as observed at lane63. Other lane sums need not have the same
    // accumulation order; broadcast that specific endpoint, not an arbitrary lane.
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x118, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x114, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x112, 0xf, 0xf, true));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x111, 0xf, 0xf, true));
    const float broadcast15 =
        __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x), __float_as_int(x), 0x142, 0xa, 0xf, true));
    x += broadcast15;
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x143, 0xf, 0xf, true));
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 63));
}
__device__ __forceinline__ float baseline_kv_sum(const float (&v)[8]) {
    // The existing producer uses four logical waves, with two adjacent
    // elements per logical lane. Compress four such lanes into one physical
    // lane while retaining the exact positive FP32 addition tree.
    float partial[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float x = v[i * 2] * v[i * 2] + v[i * 2 + 1] * v[i * 2 + 1];
        x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4e, 0xf, 0xf, false));
        x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xb1, 0xf, 0xf, false));
        partial[i] = x;
    }
    float x = (partial[0] + partial[2]) + (partial[1] + partial[3]);
    x += __shfl_xor(x, 4, 64);
    x += __shfl_xor(x, 8, 64);
    x += __shfl_xor(x, 32, 64);
    x += __shfl_xor(x, 16, 64);
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 63));
}
__device__ __forceinline__ float quad_max(float x) {
    x = fmaxf(x, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xb1, 0xf, 0xf, false)));
    return fmaxf(x, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4e, 0xf, 0xf, false)));
}
__device__ __forceinline__ float divide_q_width(float x) {
    // One compensated quotient for the fixed, exactly representable divisor.
    // Its odd factor5 keeps a finite normal exact quotient away from binary
    // halfway points. Keep native division for the subnormal/extreme domain.
    if (x >= 0x1p-114f && x <= 0x1p127f) {
        const float quotient = x * (1.f / 1280.f);
        const float residual = fmaf(-quotient, 1280.f, x);
        return fmaf(residual, 1.f / 1280.f, quotient);
    }
    return x / 1280.f;
}
__device__ __forceinline__ int scale_exponent(float maximum) {
    // Same positive-normal ceiling as the current transform producer. Its
    // amax floor prevents zero/subnormal scales for finite input activations.
    const uint32_t bits = __float_as_uint(fmaxf(maximum, 1e-4f) * (1.f / 448.f));
    return int((bits - 1) >> 23) + 1;
}
__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
    const int ab = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, ab, true);
}

__device__ __forceinline__ void query(const Args &p, int row, int lane) {
    // Emulate the loaded norm producer's 2048-value, 4-wave layout without
    // inter-wave LDS: three virtual 512-value waves (last one half valid),
    // eight contiguous values per lane, and one all-zero fourth wave.
    // This preserves its FP32 reduction tree at quantization-boundary ties.
    float values[3][8], partial[3];
#pragma unroll
    for (int stripe = 0; stripe < 3; ++stripe) {
        const int col = stripe * 512 + lane * 8;
        uint4 packed{};
        if (col < 1280)
            packed = *reinterpret_cast<const uint4 *>(p.q + int64_t(row) * p.q_stride + col);
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            values[stripe][i * 2] = bf16(uint16_t(words[i]));
            values[stripe][i * 2 + 1] = bf16(uint16_t(words[i] >> 16));
        }
        float square[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            square[i] = values[stripe][i] * values[stripe][i];
        const float even = (square[0] + square[2]) + (square[4] + square[6]);
        const float odd = (square[1] + square[3]) + (square[5] + square[7]);
        partial[stripe] = baseline_wave_sum(even + odd);
    }
    const float total = (partial[0] + partial[2]) + partial[1];
    // Division by1280 is not a power-of-two scale. The baseline emits a
    // correctly-rounded fdiv; a pre-rounded reciprocal multiply is not identical.
    const float inverse = rsqrtf(divide_q_width(total) + p.q_eps);
#pragma unroll
    for (int stripe = 0; stripe < 3; ++stripe) {
        const int col = stripe * 512 + lane * 8;
        float maximum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float weight = col < 1280 ? bf16(p.q_weight[col + i]) : 0.f;
            values[stripe][i] = (values[stripe][i] * inverse) * (weight * p.q_scale);
        }
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            rounded_pair(values[stripe][i], values[stripe][i + 1]);
            maximum = fmaxf(maximum, fabsf(values[stripe][i]));
            maximum = fmaxf(maximum, fabsf(values[stripe][i + 1]));
        }
        maximum = quad_max(maximum);
        const int exponent = scale_exponent(maximum);
        const float scale = __uint_as_float(uint32_t(254 - exponent) << 23);
        const uint2 packed = make_uint2(fp8x4(values[stripe][0] * scale, values[stripe][1] * scale,
                                              values[stripe][2] * scale, values[stripe][3] * scale),
                                        fp8x4(values[stripe][4] * scale, values[stripe][5] * scale,
                                              values[stripe][6] * scale, values[stripe][7] * scale));
        if (col < 1280)
            *reinterpret_cast<uint2 *>(p.q_out + int64_t(row) * 1280 + col) = packed;
        uint32_t sf = exponent;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 4, 64)) << 8;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 8, 64)) << 16;
        sf |= uint32_t(__shfl(exponent, (lane & ~15) + 12, 64)) << 24;
        if (col < 1280 && lane % 16 == 0)
            p.q_sf[row + int64_t(stripe * 4 + lane / 16) * p.sf_pitch] = int32_t(sf);
    }
}

__device__ __forceinline__ void prepare_swa(const Args &p, int row, int lane, float (&values)[8]) {
    const int col = lane * 8;
    const auto packed = *reinterpret_cast<const uint4 *>(p.kv + int64_t(row) * p.kv_stride + col);
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    const auto weight = *reinterpret_cast<const uint4 *>(p.kv_weight + col);
    const uint32_t weights[4] = {weight.x, weight.y, weight.z, weight.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i * 2] = bf16(uint16_t(words[i]));
        values[i * 2 + 1] = bf16(uint16_t(words[i] >> 16));
    }
    const float inverse = rsqrtf(baseline_kv_sum(values) * (1.f / 512.f) + p.kv_eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i * 2] = (values[i * 2] * inverse) * bf16(uint16_t(weights[i]));
        values[i * 2 + 1] = (values[i * 2 + 1] * inverse) * bf16(uint16_t(weights[i] >> 16));
        rounded_pair(values[i * 2], values[i * 2 + 1]);
    }
}

template <bool NEOX, bool POSITION64>
__device__ __forceinline__ void finish_swa(const Args &p, int row, int lane, float (&values)[8]) {
    const int col = lane * 8;
    const int64_t position =
        POSITION64 ? static_cast<const int64_t *>(p.positions)[row] : static_cast<const int32_t *>(p.positions)[row];
    if constexpr (NEOX) {
        // Execute shuffles on every lane, even though only the trailing RoPE
        // lanes consume them. No implicit inactive-lane source is used.
        float partner[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            partner[i] = __shfl_xor(values[i], 4, 64);
        if (lane >= 56) {
            const bool first = lane < 60;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int pair = ((col - 448) & 31) + i;
                const float c = p.cos_sin[position * p.cos_sin_stride + pair];
                const float s = p.cos_sin[position * p.cos_sin_stride + pair + 32];
                const float a = first ? values[i] : partner[i];
                const float b = first ? partner[i] : values[i];
                values[i] = first ? fmaf(a, c, -(b * s)) : fmaf(a, s, b * c);
            }
#pragma unroll
            for (int i = 0; i < 8; i += 2)
                rounded_pair(values[i], values[i + 1]);
        }
    } else if (lane >= 56) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int pair = (col - 448 + i) / 2;
            const float c = p.cos_sin[position * p.cos_sin_stride + pair];
            const float s = p.cos_sin[position * p.cos_sin_stride + pair + 32];
            const float a = values[i], b = values[i + 1];
            values[i] = fmaf(a, c, -(b * s));
            values[i + 1] = fmaf(a, s, b * c);
            rounded_pair(values[i], values[i + 1]);
        }
    }
    float maximum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        maximum = fmaxf(maximum, fabsf(values[i]));
    maximum = quad_max(maximum);
    const int exponent = scale_exponent(maximum);
    const float scale = __uint_as_float(uint32_t(254 - exponent) << 23);
    const int64_t slot = p.slots[row];
    const bool valid = slot >= 0 && slot < int64_t(p.page_size) * p.cache_blocks;
    if (valid) {
        auto *out = p.cache + (slot / p.page_size) * p.cache_block_stride + (slot % p.page_size) * p.cache_token_stride;
        *reinterpret_cast<uint2 *>(out + col) =
            make_uint2(fp8x4(values[0] * scale, values[1] * scale, values[2] * scale, values[3] * scale),
                       fp8x4(values[4] * scale, values[5] * scale, values[6] * scale, values[7] * scale));
        if (lane % 4 == 0)
            out[512 + lane / 4] = uint8_t(exponent);
    }
}

template <bool NEOX, bool POSITION64> __device__ __forceinline__ void swa(const Args &p, int row, int lane) {
    float values[8];
    prepare_swa(p, row, lane, values);
    finish_swa<NEOX, POSITION64>(p, row, lane, values);
}

template <int MODE, int ROWS, bool NEOX, bool POSITION64>
__global__ __launch_bounds__(64 * ROWS) void transform(Args p) {
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int row = blockIdx.x * ROWS + wave;
    const int lane = threadIdx.x % 64;
    if (row >= p.rows)
        return;
    if constexpr (MODE != 0)
        query(p, row, lane);
    if constexpr (MODE != 1)
        swa<NEOX, POSITION64>(p, row, lane);
}

template <int WAVES, bool NEOX, bool POSITION64> __global__ __launch_bounds__(64 * WAVES) void transform_split(Args p) {
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int row = blockIdx.x * (WAVES / 2) + wave / 2;
    if (row >= p.rows)
        return;
    if (wave % 2 == 0)
        query(p, row, threadIdx.x % 64);
    else
        swa<NEOX, POSITION64>(p, row, threadIdx.x % 64);
}

// Use physical Q waves instead of serially emulating all baseline waves in
// one wave. Only four FP32 reduction endpoints cross LDS. Every padded-row
// thread participates in the barrier; no global scratch or cross-CTA state.
template <int QW, int ROWS, bool WITH_KV, bool NEOX, bool POSITION64, bool STAGED = false>
__global__ __launch_bounds__(64 * (QW + int(WITH_KV)) * ROWS) void transform_parallel(Args p) {
    constexpr int ROW_WAVES = QW + int(WITH_KV);
    constexpr int STRIPES = (3 + QW - 1) / QW;
    __shared__ float sums[ROWS][4];
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int local_row = wave / ROW_WAVES;
    const int role = wave % ROW_WAVES;
    const int row = blockIdx.x * ROWS + local_row;
    const int lane = threadIdx.x % 64;
    const bool valid = row < p.rows;
    float values[STRIPES][8];
    uint4 weights[STRIPES];
    float kv_values[8];
    if constexpr (STAGED && WITH_KV) {
        if (valid && role == QW)
            prepare_swa(p, row, lane, kv_values);
    }
    if constexpr (QW == 3) {
        if (role == 0 && lane == 0)
            sums[local_row][3] = 0.f;
    }
    if (role < QW) {
#pragma unroll
        for (int iteration = 0; iteration < STRIPES; ++iteration) {
            const int stripe = role + iteration * QW;
            const int col = stripe * 512 + lane * 8;
            uint4 packed{};
            if (valid && col < 1280)
                packed = *reinterpret_cast<const uint4 *>(p.q + int64_t(row) * p.q_stride + col);
            if (col < 1280)
                weights[iteration] = *reinterpret_cast<const uint4 *>(p.q_weight + col);
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                values[iteration][i * 2] = bf16(uint16_t(words[i]));
                values[iteration][i * 2 + 1] = bf16(uint16_t(words[i] >> 16));
            }
            float square[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                square[i] = values[iteration][i] * values[iteration][i];
            const float even = (square[0] + square[2]) + (square[4] + square[6]);
            const float odd = (square[1] + square[3]) + (square[5] + square[7]);
            const float partial = baseline_wave_sum(even + odd);
            if (lane == 0)
                sums[local_row][stripe] = partial;
        }
    }
    __syncthreads();
    if (valid && role < QW) {
        const float total = (sums[local_row][0] + sums[local_row][2]) + (sums[local_row][1] + sums[local_row][3]);
        const float inverse = rsqrtf(divide_q_width(total) + p.q_eps);
#pragma unroll
        for (int iteration = 0; iteration < STRIPES; ++iteration) {
            const int stripe = role + iteration * QW;
            const int col = stripe * 512 + lane * 8;
            if (col < 1280) {
                const auto weight = weights[iteration];
                const uint32_t words[4] = {weight.x, weight.y, weight.z, weight.w};
                float maximum = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    values[iteration][i * 2] =
                        (values[iteration][i * 2] * inverse) * (bf16(uint16_t(words[i])) * p.q_scale);
                    values[iteration][i * 2 + 1] =
                        (values[iteration][i * 2 + 1] * inverse) * (bf16(uint16_t(words[i] >> 16)) * p.q_scale);
                    rounded_pair(values[iteration][i * 2], values[iteration][i * 2 + 1]);
                    maximum = fmaxf(maximum, fabsf(values[iteration][i * 2]));
                    maximum = fmaxf(maximum, fabsf(values[iteration][i * 2 + 1]));
                }
                maximum = quad_max(maximum);
                const int exponent = scale_exponent(maximum);
                const float scale = __uint_as_float(uint32_t(254 - exponent) << 23);
                *reinterpret_cast<uint2 *>(p.q_out + int64_t(row) * 1280 + col) =
                    make_uint2(fp8x4(values[iteration][0] * scale, values[iteration][1] * scale,
                                     values[iteration][2] * scale, values[iteration][3] * scale),
                               fp8x4(values[iteration][4] * scale, values[iteration][5] * scale,
                                     values[iteration][6] * scale, values[iteration][7] * scale));
                uint32_t sf = exponent;
                sf |= uint32_t(__shfl(exponent, (lane & ~15) + 4, 64)) << 8;
                sf |= uint32_t(__shfl(exponent, (lane & ~15) + 8, 64)) << 16;
                sf |= uint32_t(__shfl(exponent, (lane & ~15) + 12, 64)) << 24;
                if (lane % 16 == 0)
                    p.q_sf[row + int64_t(stripe * 4 + lane / 16) * p.sf_pitch] = int32_t(sf);
            }
        }
    } else if constexpr (WITH_KV) {
        if (valid && role == QW) {
            if constexpr (STAGED)
                finish_swa<NEOX, POSITION64>(p, row, lane, kv_values);
            else
                swa<NEOX, POSITION64>(p, row, lane);
        }
    }
}

template <int QW, int ROWS, bool KV, bool NEOX, bool POSITION64, bool STAGED = false>
int launch_parallel(Args args, hipStream_t stream) {
    if constexpr (ROWS * (QW + int(KV)) <= 16) {
        hipLaunchKernelGGL((transform_parallel<QW, ROWS, KV, NEOX, POSITION64, STAGED>),
                           dim3((args.rows + ROWS - 1) / ROWS), dim3(64 * ROWS * (QW + int(KV))), 0, stream, args);
        return hipGetLastError();
    }
    return hipErrorInvalidValue;
}

__global__ void empty_kernel() { asm volatile("s_nop 0"); }
__global__ void copy_kernel(const uint4 *a, uint4 *b, size_t vectors) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors; i += size_t(gridDim.x) * blockDim.x)
        b[i] = a[i];
}
__global__ void division_probe(const float *input, float *fast, float *reference, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        const float x = input[i];
        fast[i] = divide_q_width(x);
        reference[i] = x / 1280.f;
    }
}
__global__ void rounding_probe(const float *input, float *fast, float *reference, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        float a = input[i], b = -a;
        reference[i * 2] = rounded_bf16(a);
        reference[i * 2 + 1] = rounded_bf16(b);
        rounded_pair(a, b);
        fast[i * 2] = a;
        fast[i * 2 + 1] = b;
    }
}

template <int ROWS, bool NEOX, bool POSITION64> int launch_mode(int mode, Args args, hipStream_t stream) {
    if (mode == 4)
        return launch_parallel<2, ROWS, false, NEOX, POSITION64>(args, stream);
    if (mode == 5)
        return launch_parallel<2, ROWS, true, NEOX, POSITION64>(args, stream);
    if (mode == 6)
        return launch_parallel<4, ROWS, false, NEOX, POSITION64>(args, stream);
    if (mode == 7)
        return launch_parallel<4, ROWS, true, NEOX, POSITION64>(args, stream);
    if (mode == 8)
        return launch_parallel<3, ROWS, false, NEOX, POSITION64>(args, stream);
    if (mode == 9)
        return launch_parallel<3, ROWS, true, NEOX, POSITION64>(args, stream);
    if (mode == 10)
        return launch_parallel<3, ROWS, true, NEOX, POSITION64, true>(args, stream);
    if (mode == 11)
        return launch_parallel<2, ROWS, true, NEOX, POSITION64, true>(args, stream);
    if (mode == 12)
        return launch_parallel<4, ROWS, true, NEOX, POSITION64, true>(args, stream);
    const dim3 grid((args.rows + ROWS - 1) / ROWS), block(64 * ROWS);
    if (mode == 0)
        hipLaunchKernelGGL((transform<0, ROWS, NEOX, POSITION64>), grid, block, 0, stream, args);
    else if (mode == 1)
        hipLaunchKernelGGL((transform<1, ROWS, NEOX, POSITION64>), grid, block, 0, stream, args);
    else if (mode == 2)
        hipLaunchKernelGGL((transform<2, ROWS, NEOX, POSITION64>), grid, block, 0, stream, args);
    else if constexpr (ROWS >= 2) {
        if (mode != 3)
            return hipErrorInvalidValue;
        hipLaunchKernelGGL((transform_split<ROWS, NEOX, POSITION64>), dim3((args.rows + ROWS / 2 - 1) / (ROWS / 2)),
                           block, 0, stream, args);
    } else
        return hipErrorInvalidValue;
    return hipGetLastError();
}

} // namespace flash_mla::amd::post_projection

using flash_mla::amd::post_projection::Args;
extern "C" size_t post_args_size() { return sizeof(Args); }
extern "C" int post_transform(const Args *p, int mode, int rows_per_block, void *stream) {
    using namespace flash_mla::amd::post_projection;
    if (!p || p->rows < 0 || p->q_stride < 1280 || p->kv_stride < 512 || p->page_size < 1 || p->sf_pitch < p->rows)
        return hipErrorInvalidValue;
    if (!p->rows)
        return hipSuccess;
    if (rows_per_block != 1 && rows_per_block != 2 && rows_per_block != 4 && rows_per_block != 8)
        return hipErrorInvalidValue;
#define RUN(R, N, P)                                                                                                   \
    if (rows_per_block == R && bool(p->neox) == N && bool(p->positions64) == P)                                        \
        return launch_mode<R, N, P>(mode, *p, static_cast<hipStream_t>(stream));
    RUN(1, false, false)
    RUN(1, false, true) RUN(1, true, false) RUN(1, true, true) RUN(2, false, false) RUN(2, false, true)
        RUN(2, true, false) RUN(2, true, true) RUN(4, false, false) RUN(4, false, true) RUN(4, true, false)
            RUN(4, true, true) RUN(8, false, false) RUN(8, false, true) RUN(8, true, false) RUN(8, true, true)
#undef RUN
                return hipErrorInvalidValue;
}
extern "C" int post_empty(void *stream) {
    using namespace flash_mla::amd::post_projection;
    hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream));
    return hipGetLastError();
}
extern "C" int post_copy(const void *src, void *dst, size_t bytes, void *stream) {
    using namespace flash_mla::amd::post_projection;
    if (bytes % 16 || !bytes)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(copy_kernel, dim3(1024), dim3(256), 0, static_cast<hipStream_t>(stream),
                       static_cast<const uint4 *>(src), static_cast<uint4 *>(dst), bytes / 16);
    return hipGetLastError();
}
extern "C" int post_division_probe(const float *src, float *fast, float *reference, size_t count, void *stream) {
    using namespace flash_mla::amd::post_projection;
    hipLaunchKernelGGL(division_probe, dim3(1024), dim3(256), 0, static_cast<hipStream_t>(stream), src, fast, reference,
                       count);
    return hipGetLastError();
}
extern "C" int post_rounding_probe(const float *src, float *fast, float *reference, size_t count, void *stream) {
    using namespace flash_mla::amd::post_projection;
    hipLaunchKernelGGL(rounding_probe, dim3(1024), dim3(256), 0, static_cast<hipStream_t>(stream), src, fast, reference,
                       count);
    return hipGetLastError();
}
extern "C" int post_graph_nodes(void *raw) {
    size_t count = 0;
    const auto status = hipGraphGetNodes(static_cast<hipGraph_t>(raw), nullptr, &count);
    return status == hipSuccess ? int(count) : -int(status);
}
extern "C" int post_device_info(int64_t *values) {
    int device = 0;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess)
        return status;
    hipDeviceProp_t p{};
    status = hipGetDeviceProperties(&p, device);
    if (status != hipSuccess)
        return status;
    values[0] = p.memoryClockRate;
    values[1] = p.memoryBusWidth;
    values[2] = p.clockRate;
    values[3] = p.multiProcessorCount;
    values[4] = p.l2CacheSize;
    return hipSuccess;
}
