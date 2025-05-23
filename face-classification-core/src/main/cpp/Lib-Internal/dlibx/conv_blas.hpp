#pragma once

#if __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#else
#include <cblas.h>
#endif

#include <cassert>
#include <algorithm>

namespace dlibx {
    struct matrix_view {
        float const* data;
        long nrows, row_stride; ///< data + row_stride -> row 1
        long ncols, col_stride; ///< data + col_stride -> col 1
    };

    inline auto transpose(matrix_view mv) {
        std::swap(mv.nrows, mv.ncols);
        std::swap(mv.row_stride, mv.col_stride);
        return mv;
    }

    /** \brief Matrix multiply A*B and store in row major dest.
     *
     * \param dest_stride distance between rows
     */
    inline void multiply(const matrix_view& A, const matrix_view& B,
                         float* dest, long dest_stride) {
        if (A.row_stride == 1 && B.row_stride == 1 &&
            A.col_stride != 1 && B.col_stride != 1) {
            multiply(transpose(B),transpose(A),dest,dest_stride);
            return;
        }

        // output is A.nrows x B.ncols
        assert(A.ncols == B.nrows);
        assert(dest && dest_stride >= B.ncols);

        // at least one of A or B is row major
        assert(A.data && A.nrows > 0 && A.ncols > 0 &&
               A.row_stride > 0 && A.col_stride > 0 &&
               (A.row_stride == 1 || A.col_stride == 1));
        assert(B.data && B.nrows > 0 && B.ncols > 0 &&
               B.row_stride > 0 && B.col_stride > 0 &&
               (B.row_stride == 1 || B.col_stride == 1));

        auto At = CblasNoTrans, Bt = CblasNoTrans;
        auto lda = A.row_stride, ldb = B.row_stride;
        if (A.col_stride != 1) {
            assert(A.row_stride == 1);
            lda = A.col_stride;
            At = CblasTrans;
        }
        if (B.col_stride != 1) {
            assert(B.row_stride == 1);
            ldb = B.col_stride;
            Bt = CblasTrans;
        }

        cblas_sgemm(CblasRowMajor, At, Bt,
                    int(A.nrows), int(B.ncols), int(A.ncols),
                    1.0f, // alpha
                    A.data, int(lda),
                    B.data, int(ldb),
                    0.0f, // beta
                    dest, int(dest_stride));
    }
}
