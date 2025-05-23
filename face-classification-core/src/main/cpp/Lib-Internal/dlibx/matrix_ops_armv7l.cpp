
// This file is mutually exclusive among the other matrix_ops_*.cpp.
// Only link to one of them.  See matrix_ops.hpp for details.

#include "matrix_ops.hpp"
#include <arm_neon.h>

static void multiply_and_round_neon(
    int8_t* dest, const float* src, unsigned n, float multiplier) {
    static constexpr auto K = 8;
    const auto neg = vdupq_n_f32(-0.5);
    const auto pos = vdupq_n_f32(0.5);
    const auto m = vdupq_n_f32(multiplier);
    for (n = (n+(K-1))/K; n > 0; --n) {
        asm volatile("vld1.32 {d20-d23}, [%1:256]!  \n\t"
                     "vmul.f32 q10, q10, %q4   \n\t"
                     "vmul.f32 q11, q11, %q4   \n\t"
                     "vcge.f32 q12, q10, #0    \n\t"
                     "vcge.f32 q13, q11, #0    \n\t"
                     "vbsl q12, %q2, %q3       \n\t"
                     "vbsl q13, %q2, %q3       \n\t"
                     "vadd.f32 q11, q11, q13   \n\t"
                     "vadd.f32 q10, q10, q12   \n\t"
                     "vcvt.s32.f32 q12, q10    \n\t"
                     "vcvt.s32.f32 q13, q11    \n\t"
                     "vmovn.i32 d20, q12       \n\t"
                     "vmovn.i32 d21, q13       \n\t"
                     "vmovn.i16 d22, q10       \n\t"
                     "vst1.16 {d22}, [%0:64]!  \n\t"
                     : "+r"(dest), "+r"(src)
                     : "w"(pos), "w"(neg), "w"(m)
                     : "d20", "d21", "d22", "d23",
                       "d24", "d25", "d26", "d27",
                       "memory");
    }
}

static void multiply_and_round_neon(
    int16_t* dest, const float* src, unsigned n, float multiplier) {
    static constexpr auto K = 8;
    const auto neg = vdupq_n_f32(-0.5);
    const auto pos = vdupq_n_f32(0.5);
    const auto m = vdupq_n_f32(multiplier);
    for (n = (n+(K-1))/K; n > 0; --n) {
        asm volatile("vld1.32 {d20-d23}, [%1:256]!  \n\t"
                     "vmul.f32 q10, q10, %q4   \n\t"
                     "vmul.f32 q11, q11, %q4   \n\t"
                     "vcge.f32 q12, q10, #0    \n\t"
                     "vcge.f32 q13, q11, #0    \n\t"
                     "vbsl q12, %q2, %q3       \n\t"
                     "vbsl q13, %q2, %q3       \n\t"
                     "vadd.f32 q11, q11, q13   \n\t"
                     "vadd.f32 q10, q10, q12   \n\t"
                     "vcvt.s32.f32 q12, q10    \n\t"
                     "vcvt.s32.f32 q13, q11    \n\t"
                     "vmovn.i32 d20, q12       \n\t"
                     "vmovn.i32 d21, q13       \n\t"
                     "vst1.16 {d20-d21}, [%0:128]!  \n\t"
                     : "+r"(dest), "+r"(src)
                     : "w"(pos), "w"(neg), "w"(m)
                     : "d20", "d21", "d22", "d23",
                       "d24", "d25", "d26", "d27",
                       "memory");
    }
}

namespace dlibx {
    namespace ops {
        template <>
        inline int32_t
        inner_product_generic<32,int8_t,int8_t>(
            const int8_t* a, const int8_t* b, unsigned n) {
            static constexpr auto K = 32;
            int16x8_t p0, p1;
            int32x4_t s0{}, s1{};
            asm("vld1.8 {d20-d23}, [%2:256]!  \n\t"
                "vld1.8 {d24-d27}, [%3:256]!  \n\t"
                "vmull.s8 %q0, d20, d24  \n\t"
                "vmull.s8 %q1, d21, d25  \n\t"
                "vmlal.s8 %q0, d22, d26  \n\t"
                "vmlal.s8 %q1, d23, d27  \n\t"
                : "=w"(p0), "=w"(p1), "+r"(a), "+r"(b)
                :
                : "d20", "d21", "d22", "d23",
                  "d24", "d25", "d26", "d27");
            for (n = (n-1)/K; n > 0; --n) {
                asm("vld1.16 {d20-d23}, [%2:256]!  \n\t"
                    "vld1.16 {d24-d27}, [%3:256]!  \n\t"
                    "vpadal.s16 %q4, %q0  \n\t"
                    "vpadal.s16 %q5, %q1  \n\t"
                    "vmull.s8 %q0, d20, d24  \n\t"
                    "vmull.s8 %q1, d21, d25  \n\t"
                    "vmlal.s8 %q0, d22, d26  \n\t"
                    "vmlal.s8 %q1, d23, d27  \n\t"
                    : "=w"(p0), "=w"(p1), "+r"(a), "+r"(b), "+w"(s0), "+w"(s1)
                    :
                    : "d20", "d21", "d22", "d23",
                      "d24", "d25", "d26", "d27");
            }
            s0 = vpadalq_s16(s0, p0);
            s1 = vpadalq_s16(s1, p1);
            s0 = vaddq_s32(s0,s1);
            auto s = vadd_s32(vget_low_s32(s0),vget_high_s32(s0));
            return vget_lane_s32(vpadd_s32(s,s),0);
        }

        template <>
        inline int32_t
        inner_product_generic<16,int16_t,int16_t>(
            const int16_t* a, const int16_t* b, unsigned n) {
            static constexpr auto K = 16;
            int32x4_t s0, s1;
            asm("vld1.16 {d20-d23}, [%0:256]!  \n\t"
                "vld1.16 {d24-d27}, [%1:256]!  \n\t"
                //"pld [%0]  \n\t"
                //"pld [%1]  \n\t"
                "vmull.s16 %q2, d20, d24  \n\t"
                "vmull.s16 %q3, d21, d25  \n\t"
                "vmlal.s16 %q2, d22, d26  \n\t"
                "vmlal.s16 %q3, d23, d27  \n\t"
                : "+r"(a), "+r"(b), "=w"(s0), "=w"(s1)
                :
                : "d20", "d21", "d22", "d23",
                  "d24", "d25", "d26", "d27");
            for (n = (n-1)/K; n > 0; --n) {
                asm("vld1.16 {d20-d23}, [%0:256]!  \n\t"
                    "vld1.16 {d24-d27}, [%1:256]!  \n\t"
                    //"pld [%0,#64]  \n\t"
                    //"pld [%1,#64]  \n\t"
                    "vmlal.s16 %q2, d20, d24  \n\t"
                    "vmlal.s16 %q3, d21, d25  \n\t"
                    "vmlal.s16 %q2, d22, d26  \n\t"
                    "vmlal.s16 %q3, d23, d27  \n\t"
                    : "+r"(a), "+r"(b), "+w"(s0), "+w"(s1)
                    :
                    : "d20", "d21", "d22", "d23",
                      "d24", "d25", "d26", "d27");
            }
            s0 = vaddq_s32(s0,s1);
            auto s = vadd_s32(vget_low_s32(s0),vget_high_s32(s0));
            return vget_lane_s32(vpadd_s32(s,s),0);
        }

        auto inner_product_128_neon(const int8_t* a, const int8_t* b) {
            return inner_product_generic<16>(a,b,128);
        }
        auto inner_product_128_neon(const int16_t* a, const int16_t* b) {
            return inner_product_generic<16>(a,b,128);
        }

        machine_detail machine_detail::detect() {
            return {
                "armv7l_neon", 64, 8,
                &multiply_and_round_neon,
                &multiply_and_round_neon,
                &mult_row_generic<1,32>,
                &mult_row_generic<1,16>,
                inner_product_128_neon,
                inner_product_128_neon
            };
        }
    }
}
