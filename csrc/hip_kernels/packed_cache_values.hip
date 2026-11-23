// Finite E4M3 scales times half-integer E2M1 values fit exactly in BF16.
namespace flash_mla::amd::compact_attention {
using namespace flash_mla::amd::fused_sparse;
struct CacheTableWord {
    uint32_t lo0, lo1, hi0, hi1;
};
struct CacheValueTable {
    CacheTableWord words[256];
};
constexpr CacheValueTable make_cache_values() {
    CacheValueTable result{};
    constexpr int twice[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    for (int code = 0; code < 256; ++code) {
        uint32_t lo[2]{}, hi[2]{};
        const int e = (code & 127) >> 3, m = code & 7;
        for (int n = 0; n < 8; ++n) {
            const int product = (e ? 8 + m : m) * twice[n];
            uint32_t bits = (code & 128) << 8;
            if (product) {
                int leading = 0;
                for (int k = product; k > 1; k >>= 1)
                    ++leading;
                bits |= ((127 + leading + (e ? e - 11 : -10)) << 7) | ((product << (7 - leading)) & 127);
            }
            lo[n / 4] |= (bits & 255) << ((n % 4) * 8);
            hi[n / 4] |= (bits >> 8) << ((n % 4) * 8);
        }
        result.words[code] = {lo[0], lo[1], hi[0], hi[1]};
    }
    return result;
}
__device__ inline constexpr CacheValueTable cache_values = make_cache_values();

__device__ __forceinline__ U4 decode_fp4_table(uint32_t data, uint8_t scale) {
    if ((scale & 127) == 127) {
        const float nan = e4(scale);
        return U4{fp4_pair<0>(data, nan), fp4_pair<1>(data, nan), fp4_pair<2>(data, nan), fp4_pair<3>(data, nan)};
    }
    const auto table = cache_values.words[scale];
    const uint32_t even = data & 0x07070707u, odd = (data >> 4) & 0x07070707u;
    const uint32_t el = __builtin_amdgcn_perm(table.lo1, table.lo0, even);
    const uint32_t eh = __builtin_amdgcn_perm(table.hi1, table.hi0, even) ^ ((data & 0x08080808u) << 4);
    const uint32_t ol = __builtin_amdgcn_perm(table.lo1, table.lo0, odd);
    const uint32_t oh = __builtin_amdgcn_perm(table.hi1, table.hi0, odd) ^ (data & 0x80808080u);
    const uint32_t e01 = __builtin_amdgcn_perm(eh, el, 0x05010400);
    const uint32_t o01 = __builtin_amdgcn_perm(oh, ol, 0x05010400);
    const uint32_t e23 = __builtin_amdgcn_perm(eh, el, 0x07030602);
    const uint32_t o23 = __builtin_amdgcn_perm(oh, ol, 0x07030602);
    return U4{__builtin_amdgcn_perm(o01, e01, 0x05040100), __builtin_amdgcn_perm(o01, e01, 0x07060302),
        __builtin_amdgcn_perm(o23, e23, 0x05040100), __builtin_amdgcn_perm(o23, e23, 0x07060302)};
}
template <int Byte> __device__ __forceinline__ uint32_t decode_fp4_pair(uint32_t data, float scale) {
#ifdef FLASH_MLA_AMD_FP4_HALF_PAIR
    using H2 = _Float16 __attribute__((ext_vector_type(2)));
    const H2 raw = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(data, 1.f, Byte);
    const _Float16 factor = static_cast<_Float16>(scale);
    // Finite E4M3 * finite E2M1 is exact in binary16 (at most 8 significant bits).
    const H2 product = raw * H2{factor, factor};
    F2 widened = __builtin_convertvector(product, F2);
    float lo = widened[0], hi = widened[1];
    asm volatile("" : "+v"(lo), "+v"(hi));
    return pack2(lo, hi);
#elif defined(FLASH_MLA_AMD_FP4_FLOAT_PAIR)
    const F2 value = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(data, 1.f, Byte);
    return pack2(value[0] * scale, value[1] * scale);
#else
    return fp4_pair<Byte>(data, scale);
#endif
}
__device__ __forceinline__ uint32_t decode_fp8_pair(uint32_t data, uint8_t scale) {
    const float factor = ue8(scale);
    if (scale < 16 || scale > 239) {
        const auto z = __builtin_amdgcn_cvt_pk_f32_fp8(data, false);
        return pack2(z[0] * factor, z[1] * factor);
    }
    return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(data, factor, 0));
}
} // namespace flash_mla::amd::compact_attention
