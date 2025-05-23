
// See matrix_ops.hpp for details regarding the matrix_ops_*.cpp files.

// these methods require AVX2, and will only be used
// if matrix_ops_x86_detect.cpp is included

#include "matrix_ops.hpp"
#include <immintrin.h>

using namespace dlibx;

template <>
void ops::multiply_and_round_avx<int8_t>(
    int8_t* dest, const float* src, unsigned n, float multiplier) {
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    auto d = reinterpret_cast<__m128i*>(dest);
    const auto m = _mm256_set1_ps(multiplier);
    for (n = (n+15)/16; n > 0; --n) {
        const auto x0 = _mm256_mul_ps(m,_mm256_load_ps(src));  src += 8;
        const auto x1 = _mm256_mul_ps(m,_mm256_load_ps(src));  src += 8;
        const auto y0 = _mm256_cvtps_epi32(x0);
        const auto y1 = _mm256_cvtps_epi32(x1);
        const auto i16 =
            _mm256_permute4x64_epi64(_mm256_packs_epi32(y0,y1),0xd8);
        *d++ = _mm_packs_epi16(_mm256_castsi256_si128(i16),
                               _mm256_extracti128_si256(i16,1));
    }
}

template <>
void ops::multiply_and_round_avx<int16_t>(
    int16_t* dest, const float* src, unsigned n, float multiplier) {
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    auto d = reinterpret_cast<__m256i*>(dest);
    const auto m = _mm256_set1_ps(multiplier);
    for (n = (n+15)/16; n > 0; --n) {
        const auto x0 = _mm256_mul_ps(m,_mm256_load_ps(src));  src += 8;
        const auto x1 = _mm256_mul_ps(m,_mm256_load_ps(src));  src += 8;
        const auto y0 = _mm256_cvtps_epi32(x0);
        const auto y1 = _mm256_cvtps_epi32(x1);
        *d++ = _mm256_permute4x64_epi64(_mm256_packs_epi32(y0,y1),0xd8);
    }
}

template <unsigned N = 0>
static inline __m128i
inner_product(const int8_t* _ap, const int8_t* _bp, unsigned n = N) {
    auto a256i = reinterpret_cast<const __m256i*>(_ap);
    auto b256i = reinterpret_cast<const __m256i*>(_bp);

    const __m256i one = _mm256_set1_epi16(1);

    auto a0 = _mm256_lddqu_si256(a256i);
    auto b0 = _mm256_lddqu_si256(b256i);

    if (n <= 32) {
        auto s0 = _mm256_sign_epi8(a0,b0);
        auto u0 = _mm256_abs_epi8(b0);
        auto m0 = _mm256_maddubs_epi16(u0,s0);
        auto y0 = _mm256_madd_epi16(m0,one); // -> 8xi32
        auto z0 = _mm_add_epi32(_mm256_castsi256_si128(y0),
                                _mm256_extracti128_si256(y0,1));
        z0 = _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x4e));
        return _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x11));
        //z0 = _mm_hadd_epi32(z0,z0);
        //return _mm_hadd_epi32(z0,z0);
    }

    auto a1 = _mm256_lddqu_si256(++a256i);
    auto b1 = _mm256_lddqu_si256(++b256i);

    auto s0 = _mm256_sign_epi8(a0,b0);
    auto u0 = _mm256_abs_epi8(b0);

    n = (n-33)/32;
    if (n <= 0) {
        const auto s1 = _mm256_sign_epi8(a1,b1);
        const auto u1 = _mm256_abs_epi8(b1);
        const auto m0 = _mm256_maddubs_epi16(u0,s0); // -> 16xi16
        const auto m1 = _mm256_maddubs_epi16(u1,s1);
        const auto y0 = _mm256_madd_epi16(m0,one);   // -> 8xi32
        const auto y1 = _mm256_madd_epi16(m1,one);
        const auto w0 = _mm256_add_epi32(y0,y1);
        auto z0 = _mm_add_epi32(_mm256_castsi256_si128(w0),
                                _mm256_extracti128_si256(w0,1)); // -> 4xi32
        z0 = _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x4e));   // -> 2xi32
        return _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x11)); // -> 1xi32
    }

    // assert(n > 0);
    // s0/u0 and a1/b1 are active

    a0 = _mm256_lddqu_si256(++a256i);
    b0 = _mm256_lddqu_si256(++b256i);

    auto s1 = _mm256_sign_epi8(a1,b1);
    auto u1 = _mm256_abs_epi8(b1);

    // s0/u0, s1/u1 and a0/b0 are active

    const auto m0 = _mm256_maddubs_epi16(u0,s0); // -> 16xi16
    const auto m1 = _mm256_maddubs_epi16(u1,s1);

    auto w0 = _mm256_madd_epi16(m0,one);   // -> 8xi32
    auto w1 = _mm256_madd_epi16(m1,one);

    s0 = _mm256_sign_epi8(a0,b0);
    u0 = _mm256_abs_epi8(b0);

    while (--n > 0) {
        // only s0/u0 are active at start of loop

        a1 = _mm256_lddqu_si256(++a256i);
        b1 = _mm256_lddqu_si256(++b256i);

        if (--n == 0) {
            const auto m0 = _mm256_maddubs_epi16(u0,s0); // -> 16xi16
            s0 = _mm256_sign_epi8(a1,b1);
            const auto y0 = _mm256_madd_epi16(m0,one);   // -> 8xi32
            u0 = _mm256_abs_epi8(b1);
            w0 = _mm256_add_epi32(w0,y0);
            break; // note: a1/b1 got moved to s0/u0
        }

        a0 = _mm256_lddqu_si256(++a256i);
        b0 = _mm256_lddqu_si256(++b256i);

        s1 = _mm256_sign_epi8(a1,b1);
        u1 = _mm256_abs_epi8(b1);

        // s0/u0, s1/u1 and a0/b0 are active

        const auto m0 = _mm256_maddubs_epi16(u0,s0); // -> 16xi16
        const auto m1 = _mm256_maddubs_epi16(u1,s1); // -> 16xi16
        const auto y0 = _mm256_madd_epi16(m0,one);   // -> 8xi32
        const auto y1 = _mm256_madd_epi16(m1,one);   // -> 8xi32
        s0 = _mm256_sign_epi8(a0,b0);
        u0 = _mm256_abs_epi8(b0);
        w0 = _mm256_add_epi32(w0,y0);
        w1 = _mm256_add_epi32(w1,y1);
    }

    // only s0/u0 are active
    const auto m2 = _mm256_maddubs_epi16(u0,s0); // -> 16xi16
    w0 = _mm256_add_epi32(w0,w1);
    const auto y2 = _mm256_madd_epi16(m2,one);   // -> 8xi32
    w0 = _mm256_add_epi32(w0,y2);

    auto z0 = _mm_add_epi32(_mm256_castsi256_si128(w0),
                           _mm256_extracti128_si256(w0,1));
    z0 = _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x4e));
    return _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x11));
    //z0 = _mm_hadd_epi32(z0,z0);
    //return _mm_hadd_epi32(z0,z0);
}

int32_t ops::inner_product_128_i8_avx(const int8_t* a, const int8_t* b) {
    return _mm_cvtsi128_si32(inner_product<128>(a, b));
}

template <unsigned N = 0>
static inline __m128i
inner_product(const int16_t* a, const int16_t* b, unsigned n = N) {
    auto a256i = reinterpret_cast<const __m256i*>(a);
    auto b256i = reinterpret_cast<const __m256i*>(b);
    auto sum = _mm256_madd_epi16(*a256i, *b256i);
    auto x1 = _mm256_madd_epi16(*++a256i, *++b256i);
    for (n = (n-1)/32; n > 0; --n) {
        auto x0 = _mm256_madd_epi16(*++a256i, *++b256i);
        sum = _mm256_add_epi32(sum, x1);
        x1 = _mm256_madd_epi16(*++a256i, *++b256i);
        sum = _mm256_add_epi32(sum, x0);
    }
    sum = _mm256_add_epi32(sum, x1);
    auto s = _mm_add_epi32(_mm256_castsi256_si128(sum),
                           _mm256_extracti128_si256(sum,1));
    s = _mm_hadd_epi32(s,s);
    s = _mm_hadd_epi32(s,s);
    return s;
}

int32_t ops::inner_product_128_i16_avx(const int16_t* a, const int16_t* b) {
    return _mm_cvtsi128_si32(inner_product<128>(a, b));
}

template <typename T>
void ops::mult_row_avx(
    float* dest, float lhs_coeff, const T* lhs_value, unsigned nvals,
    const float* rhs_coeff, const T* rhs_value, unsigned rhs_stride,
    unsigned n) {

    const auto lc = _mm_set1_ps(lhs_coeff);
    for (n = (n+3)/4; n > 0; --n, dest += 4, rhs_coeff += 4) {
        auto f4 = _mm_mul_ps(lc,_mm_load_ps(rhs_coeff));
        const auto fb = _mm_movemask_ps(_mm_cmpneq_ps(f4,_mm_set1_ps(0)));
        __m128i s0{}, s1{};
        if (fb&1)
            s0 = ::inner_product(lhs_value, rhs_value, nvals);
        rhs_value += rhs_stride;
        if (fb&2)
            s1 = ::inner_product(lhs_value, rhs_value, nvals);
        rhs_value += rhs_stride;
        if (fb&4)
            s0 = _mm_unpacklo_epi32(
                s0, ::inner_product(lhs_value, rhs_value, nvals));
        rhs_value += rhs_stride;
        if (fb&8)
            s1 = _mm_unpacklo_epi32(
                s1, ::inner_product(lhs_value, rhs_value, nvals));
        rhs_value += rhs_stride;
        f4 = _mm_mul_ps(f4,_mm_cvtepi32_ps(_mm_unpacklo_epi32(s0,s1)));
        _mm_store_ps(dest, f4);
    }
}

// explicit template instantiation
template void ops::mult_row_avx(float*, float, const int8_t*, unsigned, const float*, const int8_t*, unsigned, unsigned);
template void ops::mult_row_avx(float*, float, const int16_t*, unsigned, const float*, const int16_t*, unsigned, unsigned);
