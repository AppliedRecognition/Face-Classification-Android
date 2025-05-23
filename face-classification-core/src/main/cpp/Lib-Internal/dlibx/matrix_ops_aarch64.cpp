
// This file is mutually exclusive among the other matrix_ops_*.cpp.
// Only link to one of them.  See matrix_ops.hpp for details.

#include "matrix_ops.hpp"
#include <arm_neon.h>

static inline auto
inner_product_neon(const int8_t* p0, const int8_t* p1, unsigned n) {
    static constexpr auto K = 32;
    int32x4_t s0{}, s1{};
    int16x8_t h0, h1;
    int8x16_t a0, b0, a1, b1;
    asm("ldr %q0, [%4]  \n\t"
        "ldr %q1, [%5]  \n\t"
        "ldr %q2, [%4,16]!  \n\t"
        "ldr %q3, [%5,16]!  \n\t"
        : "=w"(a0), "=w"(b0), "=w"(a1), "=w"(b1),
          "+r"(p0), "+r"(p1)
        : : );
    for (n = (n-1)/K; n > 0; --n) {
        asm("smull  %2.8h, %4.8b,  %5.8b  \n\t"
            "smull2 %3.8h, %4.16b, %5.16b  \n\t"
            "ldr %q4, [%8,16]  \n\t"
            "ldr %q5, [%9,16]  \n\t"
            "smlal  %2.8h, %6.8b,  %7.8b  \n\t"
            "smlal2 %3.8h, %6.16b, %7.16b  \n\t"
            "ldr %q6, [%8,32]!  \n\t"
            "ldr %q7, [%9,32]!  \n\t"
            "sadalp %0.4s, %2.8h  \n\t"
            "sadalp %1.4s, %3.8h  \n\t"
            : "+w"(s0), "+w"(s1), "=w"(h0), "=w"(h1),
              "+w"(a0), "+w"(b0), "+w"(a1), "+w"(b1),
              "+r"(p0), "+r"(p1)
            : : );
    }
    asm("smull  %2.8h, %4.8b,  %5.8b  \n\t"
        "smull2 %3.8h, %4.16b, %5.16b  \n\t"
        "smlal  %2.8h, %6.8b,  %7.8b  \n\t"
        "smlal2 %3.8h, %6.16b, %7.16b  \n\t"
        "sadalp %0.4s, %2.8h  \n\t"
        "sadalp %1.4s, %3.8h  \n\t"
        : "+w"(s0), "+w"(s1), "=w"(h0), "=w"(h1),
          "+w"(a0), "+w"(b0), "+w"(a1), "+w"(b1)
        : : );
    return vaddvq_s32(vaddq_s32(s0,s1));
}

static auto inner_product_128(const int8_t* p0, const int8_t* p1) {
    return inner_product_neon(p0, p1, 128);
}

static inline auto
inner_product_neon(const int16_t* p0, const int16_t* p1, unsigned n) {
    static constexpr auto K = 16;
    int32x4_t s0{}, s1{};
    int16x8_t a0, b0, a1, b1;
    asm("ldr %q0, [%4]  \n\t"
        "ldr %q1, [%5]  \n\t"
        "ldr %q2, [%4,16]!  \n\t"
        "ldr %q3, [%5,16]!  \n\t"
        : "=w"(a0), "=w"(b0), "=w"(a1), "=w"(b1),
          "+r"(p0), "+r"(p1)
        : : );
    for (n = (n-1)/K; n > 0; --n) {
        asm("smlal  %4.4s, %0.4h, %1.4h  \n\t"
            "smlal2 %5.4s, %0.8h, %1.8h  \n\t"
            "ldr %q0, [%6,16]  \n\t"
            "ldr %q1, [%7,16]  \n\t"
            "smlal  %4.4s, %2.4h, %3.4h  \n\t"
            "smlal2 %5.4s, %2.8h, %3.8h  \n\t"
            "ldr %q2, [%6,32]!  \n\t"
            "ldr %q3, [%7,32]!  \n\t"
            : "+w"(a0), "+w"(b0), "+w"(a1), "+w"(b1),
              "+w"(s0), "+w"(s1), "+r"(p0), "+r"(p1)
            : : );
    }
    asm("smlal  %4.4s, %0.4h, %1.4h  \n\t"
        "smlal2 %5.4s, %0.8h, %1.8h  \n\t"
        "smlal  %4.4s, %2.4h, %3.4h  \n\t"
        "smlal2 %5.4s, %2.8h, %3.8h  \n\t"
        : "+w"(a0), "+w"(b0), "+w"(a1), "+w"(b1),
          "+w"(s0), "+w"(s1)
        : : );
    return vaddvq_s32(vaddq_s32(s0,s1));
}

static auto inner_product_128(const int16_t* p0, const int16_t* p1) {
    return inner_product_neon(p0, p1, 128);
}

template <typename T>
static void mult_row_neon(
    float* _dest, float lhs_coeff, const T* lhs_value, unsigned nvals,
    const float* _rhs_coeff, const T* rhs_value, unsigned rhs_stride,
    unsigned n) {

    auto dest = reinterpret_cast<float32x4_t*>(_dest);
    auto rhs_coeff = reinterpret_cast<const float32x4_t*>(_rhs_coeff);

    const auto lc = vdupq_n_f32(lhs_coeff);

    const auto mask = uint32x4_t{1,2,4,8};

    int32x4_t s{};
    for (n = (n+3)/4; n > 0; --n, ++dest, ++rhs_coeff) {
        auto f4 = vmulq_f32(*rhs_coeff, lc);
        const auto fb = vaddvq_u32(vandq_u32(vceqzq_f32(f4),mask));
        if (!(fb&1))
            s[0] = inner_product_neon(lhs_value, rhs_value, nvals);
        rhs_value += rhs_stride;
        if (!(fb&2))
            s[1] = inner_product_neon(lhs_value, rhs_value, nvals);
        rhs_value += rhs_stride;
        if (!(fb&4))
            s[2] = inner_product_neon(lhs_value, rhs_value, nvals);
        rhs_value += rhs_stride;
        if (!(fb&8))
            s[3] = inner_product_neon(lhs_value, rhs_value, nvals);
        rhs_value += rhs_stride;
        *dest = vmulq_f32(f4, vcvtq_f32_s32(s));
    }
}

using namespace dlibx::ops;

machine_detail machine_detail::detect() {
    return {
        "aarch64", 128, 8,
        &multiply_and_round_generic<16>,
        &multiply_and_round_generic<16>,
        &mult_row_neon,
        &mult_row_neon,
        &inner_product_128,
        &inner_product_128
    };
}

