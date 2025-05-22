#pragma once

#include <cstdint>
#include <cmath>
#include <stdext/rounding.hpp>

/** \brief Optimized matrix operations for qmat.
 *
 * How to compile:
 *
 * On the ARM platform select one and only one of the following:
 *   matrix_ops_armv7l.cpp  -- arm-32bit NEON
 *   matrix_ops_aarch64.cpp -- arm-64bit optimized
 *   matrix_ops_generic.cpp -- non-optimized version for any platform
 *
 * On X86, select either:
 *   matrix_ops_x86_sse.cpp -- SSE2 implementation
 *   matrix_ops_x86_sse_only.cpp
 * or:
 *   matrix_ops_x86_sse.cpp -- SSE2 implementation
 *   matrix_ops_x86_avx.cpp -- AVX2 implementation
 *   matrix_ops_x86_detect.cpp -- SSE2 or AVX2 selected at runtime
 * In this latter case, all files except matrix_ops_x86_avx.cpp should be
 * compiled without avx, while this one file must be compiled with avx2 enabled.
 */

namespace dlibx {
    namespace ops {

        /** \brief Methods detected at runtime for the current platform.
         *
         * If max_8bit_bits < 8, then the int8_t version of inner product
         * either doesn't work in the full 8-bit times 8-bit case, or is slow.
         * Specifically, if max_8bit_bits == 7, then at least one side
         * must be limited to 7-bit values in the range [64,64).
         */
        struct machine_detail {
            const char* description;
            std::size_t cache_kb;
            unsigned max_8bit_bits; // max lhs bits safe to use with 8bit method

            void(*multiply_and_round_i8)(int8_t*,const float*,unsigned,float);
            void(*multiply_and_round_i16)(int16_t*,const float*,unsigned,float);

            void(*mult_row_i8)(float*,float,const int8_t*,unsigned,
                               const float*,const int8_t*,unsigned,unsigned);
            void(*mult_row_i16)(float*,float,const int16_t*,unsigned,
                                const float*,const int16_t*,unsigned,unsigned);

            // for 128 element template comparison
            int32_t(*inner_product_128_i8)(const int8_t*, const int8_t*);
            int32_t(*inner_product_128_i16)(const int16_t*, const int16_t*);

            static machine_detail detect();
        };
        extern const machine_detail machine;

        /** \brief Multiply each element from src by the coeff and round.
         *
         * Both dest and src buffers must be an integer multiple of 64 bytes.
         */
        inline void multiply_and_round(
            int8_t* dest, const float* src, unsigned n, float coeff) {
            machine.multiply_and_round_i8(dest, src, n, coeff);
        }
        inline void multiply_and_round(
            int16_t* dest, const float* src, unsigned n, float coeff) {
            machine.multiply_and_round_i16(dest, src, n, coeff);
        }

        /** \brief Perform multiple inner products and multiply coefficients.
         *
         * For each of the n rows from rhs, multiply lhs_coeff by rhs_coeff
         * by inner_product(lhs_value,rhs_value) and store result in dest.
         *
         * Each value buffers must be an integer multiple of 64 bytes and
         * at least one side must have zeros stored in the extra bytes.
         * The rhs_coeff buffer must be an integer multiple of 16 bytes and
         * the extra bytes should have zero stored.
         * The dest buffer must also be an integer multiple of 16 bytes. 
         */
        inline void mult_row(
            float* dest,
            float lhs_coeff, const int8_t* lhs_value, unsigned nvals,
            const float* rhs_coeff,
            const int8_t* rhs_value, unsigned rhs_stride,
            unsigned n) {
            machine.mult_row_i8(dest, lhs_coeff, lhs_value, nvals,
                                rhs_coeff, rhs_value, rhs_stride, n);
        }
        inline void mult_row(
            float* dest,
            float lhs_coeff, const int16_t* lhs_value, unsigned nvals,
            const float* rhs_coeff,
            const int16_t* rhs_value, unsigned rhs_stride,
            unsigned n) {
            machine.mult_row_i16(dest, lhs_coeff, lhs_value, nvals,
                                 rhs_coeff, rhs_value, rhs_stride, n);
        }
        

        /** \brief Multiply each src element by multiplier and round to int16.
         *
         * Generic version works on any platform.
         *
         * Higher values of L allow for compiler vectorization, but 
         * rows must be padded so as to allow run past end of data.
         * L should be power of 2. 
         * For 64 byte cache line, maximum L is 16.
         */
        template <unsigned L = 4, typename T = int16_t>
        void multiply_and_round_generic(
            T* dest, const float* src, unsigned n, float multiplier) {
            for (n = (n+(L-1))/L; n > 0; --n)
                for (auto l = L; l > 0; --l, ++dest, ++src)
                    *dest = T(stdx::round(multiplier * *src));
        }

        /** \brief Fixed point inner product.
         *
         * Generic version works on any platform.
         *
         * Higher values of K allow for compiler vectorization, but 
         * rows must be padded so as to allow run past end of data.
         * K should be power of 2. 
         * For 64 byte cache line, maximum K is 32 (int16_t) or 64 (int8_t).
         */
        template <unsigned K, typename TA = int16_t, typename TB = TA>
        inline int32_t
        inner_product_generic(const TA* a, const TB* b, unsigned n) {
            int32_t sum = 0;
            for (n = (n+(K-1))/K; n > 0; --n)
                for (auto k = K; k > 0; --k, ++a, ++b)
                    sum += *a * *b;
            return sum;
        }

        inline bool not_zero(float x) {
            return x < 0 || x > 0;
        }
        
        /** \brief Matrix multiply single lhs row by n rhs rows to
         * product n columns in dest.
         *
         * Generic version works on any platform.
         */
        template <unsigned L = 4, unsigned K = 4,
                  typename LHS = int16_t, typename RHS = LHS>
        void mult_row_generic(
            float* dest,
            float lhs_coeff, const LHS* lhs_value, unsigned nvals,
            const float* rhs_coeff,
            const RHS* rhs_value, unsigned rhs_stride,
            unsigned n) {

            for (n = (n+(L-1))/L; n > 0; --n) {
                for (unsigned i = 0; i < L; ++i, ++rhs_coeff)
                    dest[i] = lhs_coeff * *rhs_coeff;
                for (auto l = L; l > 0; --l, ++dest, rhs_value += rhs_stride)
                    if (not_zero(*dest))
                        *dest *= float(inner_product_generic<K>(
                                           lhs_value, rhs_value, nvals));
            }
        }


        /// only available if matrix_ops_x86_sse.cpp is included
        template <typename T>
        void multiply_and_round_sse(
            T* dest, const float* src, unsigned n, float multiplier);
        template <typename T>
        void mult_row_sse(
            float* dest,
            float lhs_coeff, const T* lhs_value, unsigned nvals,
            const float* rhs_coeff, const T* rhs_value, unsigned stride,
            unsigned n);
        int32_t inner_product_128_i8_sse(const int8_t*, const int8_t*);
        int32_t inner_product_128_i16_sse(const int16_t*, const int16_t*);

        /// only available if matrix_ops_x86_avx.cpp is included
        template <typename T>
        void multiply_and_round_avx(
            T* dest, const float* src, unsigned n, float multiplier);
        template <typename T>
        void mult_row_avx(
            float* dest,
            float lhs_coeff, const T* lhs_value, unsigned nvals,
            const float* rhs_coeff, const T* rhs_value, unsigned stride,
            unsigned n);
        int32_t inner_product_128_i8_avx(const int8_t*, const int8_t*);
        int32_t inner_product_128_i16_avx(const int16_t*, const int16_t*);
    }
}
