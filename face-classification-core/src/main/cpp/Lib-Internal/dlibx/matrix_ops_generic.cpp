
// This file is mutually exclusive among the other matrix_ops_*.cpp.
// Only link to one of them.  See matrix_ops.hpp for details.

#include "matrix_ops.hpp"

using namespace dlibx::ops;

// note: expect these methods to be slow as they are
// the reference implementation
// it's better to use the platform specific implementations

static auto inner_product_128(const int8_t* a, const int8_t* b) {
    return inner_product_generic<128>(a,b,128);
}
static auto inner_product_128(const int16_t* a, const int16_t* b) {
    return inner_product_generic<128>(a,b,128);
}

machine_detail machine_detail::detect() {
    return {
        "generic", 64, 8,
        &multiply_and_round_generic<1>,
        &multiply_and_round_generic<1>,
        &mult_row_generic<1,1>,
        &mult_row_generic<1,1>,
        &inner_product_128,
        &inner_product_128
    };
}
