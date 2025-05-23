
// See matrix_ops.hpp for details regarding the matrix_ops_*.cpp files.

// these methods require SSE2, but SSE2 is manditory on all AMD64 platforms

#include "matrix_ops.hpp"
#include <immintrin.h>

using namespace dlibx;

template <>
void ops::multiply_and_round_sse<int8_t>(
    int8_t* dest, const float* src, unsigned n, float multiplier) {
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    auto d = reinterpret_cast<__m128i*>(dest);
    const auto m = _mm_set1_ps(multiplier);
    auto x0 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
    auto x1 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
    auto x2 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
    auto x3 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
    for (n = (n-1)/16; n > 0; --n) {
        auto y0 = _mm_packs_epi32(_mm_cvtps_epi32(x0),_mm_cvtps_epi32(x1));
        x0 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
        x1 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
        auto y1 = _mm_packs_epi32(_mm_cvtps_epi32(x2),_mm_cvtps_epi32(x3));
        x2 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
        x3 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
        *d++ = _mm_packs_epi16(y0,y1);
    }
    auto y0 = _mm_packs_epi32(_mm_cvtps_epi32(x0),_mm_cvtps_epi32(x1));
    auto y1 = _mm_packs_epi32(_mm_cvtps_epi32(x2),_mm_cvtps_epi32(x3));
    *d++ = _mm_packs_epi16(y0,y1);
}

template <>
void ops::multiply_and_round_sse<int16_t>(
    int16_t* dest, const float* src, unsigned n, float multiplier) {
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    auto d = reinterpret_cast<__m128i*>(dest);
    const auto m = _mm_set1_ps(multiplier);
    auto x0 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
    auto x1 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
    auto x2 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
    auto x3 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
    *d++ = _mm_packs_epi32(_mm_cvtps_epi32(x0),_mm_cvtps_epi32(x1));
    for (n = (n-1)/16; n > 0; --n) {
        x0 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
        x1 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
        *d++ = _mm_packs_epi32(_mm_cvtps_epi32(x2),_mm_cvtps_epi32(x3));
        x2 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
        x3 = _mm_mul_ps(m,_mm_load_ps(src));  src += 4;
        *d++ = _mm_packs_epi32(_mm_cvtps_epi32(x0),_mm_cvtps_epi32(x1));
    }
    *d++ = _mm_packs_epi32(_mm_cvtps_epi32(x2),_mm_cvtps_epi32(x3));
}

static inline __m128i
inner_product(const int8_t* _ap, const int8_t* _bp, unsigned n) {
    auto a128 = reinterpret_cast<const __m128i*>(_ap);
    auto b128 = reinterpret_cast<const __m128i*>(_bp);

    const __m128i one = _mm_set1_epi16(1);

    auto a0 = _mm_lddqu_si128(a128);
    auto b0 = _mm_lddqu_si128(b128);

    if (n <= 16) {
        auto s0 = _mm_sign_epi8(a0,b0);
        auto u0 = _mm_abs_epi8(b0);
        auto m0 = _mm_maddubs_epi16(u0,s0);
        auto z0 = _mm_madd_epi16(m0,one); // -> 4xi32
        //z0 = _mm_hadd_epi32(z0,z0);
        //return _mm_hadd_epi32(z0,z0);
        z0 = _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x4e));
        return _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x11));
    }

    auto a1 = _mm_lddqu_si128(++a128);
    auto b1 = _mm_lddqu_si128(++b128);

    auto s0 = _mm_sign_epi8(a0,b0);
    auto u0 = _mm_abs_epi8(b0);
    auto m0 = _mm_maddubs_epi16(u0,s0);
    auto z0 = _mm_madd_epi16(m0,one); // -> 4xi32

    for (n = (n-17)/16; n > 0; --n) {
        auto s1 = _mm_sign_epi8(a1,b1);
        auto u1 = _mm_abs_epi8(b1);

        a1 = _mm_lddqu_si128(++a128);
        b1 = _mm_lddqu_si128(++b128);

        auto m1 = _mm_maddubs_epi16(u1,s1);
        auto z1 = _mm_madd_epi16(m1,one);
        z0 = _mm_add_epi32(z0,z1);
    }

    auto s1 = _mm_sign_epi8(a1,b1);
    auto u1 = _mm_abs_epi8(b1);
    auto m1 = _mm_maddubs_epi16(u1,s1);
    auto z1 = _mm_madd_epi16(m1,one);
    z0 = _mm_add_epi32(z0,z1);

    //z0 = _mm_hadd_epi32(z0,z0);
    //return _mm_hadd_epi32(z0,z0);
    z0 = _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x4e));
    return _mm_add_epi32(z0, _mm_shuffle_epi32(z0,0x11));
}

int32_t ops::inner_product_128_i8_sse(const int8_t* a, const int8_t* b) {
    return _mm_cvtsi128_si32(inner_product(a, b, 128));
}

static inline __m128i
inner_product(const int16_t* a, const int16_t* b, unsigned n) {
    auto a128i = reinterpret_cast<const __m128i*>(a);
    auto b128i = reinterpret_cast<const __m128i*>(b);
    auto sum = _mm_madd_epi16(*a128i, *b128i);
    auto x1 = _mm_madd_epi16(*++a128i, *++b128i);
    for (n = (n-1)/16; n > 0; --n) {
        auto x0 = _mm_madd_epi16(*++a128i, *++b128i);
        sum = _mm_add_epi32(sum, x1);
        x1 = _mm_madd_epi16(*++a128i, *++b128i);
        sum = _mm_add_epi32(sum, x0);
    }
    sum = _mm_add_epi32(sum, x1);
    //sum = _mm_hadd_epi32(sum,sum);  // hadd is slow and requires SSSE3
    //sum = _mm_hadd_epi32(sum,sum);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum,0x4e));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum,0x11));
    return sum;
}

int32_t ops::inner_product_128_i16_sse(const int16_t* a, const int16_t* b) {
    return _mm_cvtsi128_si32(inner_product(a, b, 128));
}

template <typename T>
void ops::mult_row_sse(
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
template void ops::mult_row_sse(float*, float, const int8_t*, unsigned, const float*, const int8_t*, unsigned, unsigned);
template void ops::mult_row_sse(float*, float, const int16_t*, unsigned, const float*, const int16_t*, unsigned, unsigned);
