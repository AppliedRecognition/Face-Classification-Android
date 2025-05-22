
// This file is mutually exclusive with the matrix_ops_x86_sse_only.cpp file.
// Both matrix_ops_x86_sse.cpp and matrix_ops_x86_avx.cpp must be included
// with this file.

#include "matrix_ops.hpp"

using namespace dlibx::ops;

/*
#define OSXSAVEFlag (1<<27)
#define AVXFlag     ((1<<28)|OSXSAVEFlag)
static inline bool SimdDetectFeature(int idFeature) {
    int EAX, EBX, ECX, EDX;
    cpuid(0, EAX, EBX, ECX, EDX);
    return (ECX & idFeature) == idFeature;
}
*/

static inline bool avx2_available() {
#if defined(_MSC_VER)
    return true;
#else
    return __builtin_cpu_supports("avx2");
#endif
}

machine_detail machine_detail::detect() {
    if (avx2_available())
        return { "AVX2", 256, 7, ///< calc is wrong at 8-bit so 7 bit max
                 &multiply_and_round_avx,
                 &multiply_and_round_avx,
                 &mult_row_avx,
                 &mult_row_avx,
                 &inner_product_128_i8_avx,
                 &inner_product_128_i16_avx };
    else
        return { "SSE2", 256, 0, ///< 8-bit method is a lot slower than 16-bit
                 &multiply_and_round_sse,
                 &multiply_and_round_sse,
                 &mult_row_sse,
                 &mult_row_sse,
                 &inner_product_128_i8_sse,
                 &inner_product_128_i16_sse };
}
