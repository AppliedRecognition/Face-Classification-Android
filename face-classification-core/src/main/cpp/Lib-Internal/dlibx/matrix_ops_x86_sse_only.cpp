
// This file is mutually exclusive with the matrix_ops_x86_detect.cpp file.
// The file matrix_ops_x86_sse.cpp must be included with this file.

#include "matrix_ops.hpp"

using namespace dlibx::ops;

machine_detail machine_detail::detect() {
    return { "SSE2", 256, 0, ///< 8-bit method is a lot slower than 16-bit
             &multiply_and_round_sse,
             &multiply_and_round_sse,
             &mult_row_sse,
             &mult_row_sse,
             &inner_product_128_i8_sse,
             &inner_product_128_i16_sse };
}

