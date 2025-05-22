#pragma once

#include <dlib/matrix/matrix_exp.h>
#include <stdext/aligned_alloc.hpp>
#include <algorithm>

namespace dlibx {
    template <typename T, unsigned cache_line_bytes>
    class aligned_matrix;
}

namespace dlib {
    template <typename T, unsigned cache_line_bytes>
    struct matrix_traits<dlibx::aligned_matrix<T,cache_line_bytes> > {
        using type = T;
        using const_ret_type = const T&;
        using layout_type = row_major_layout;
        using mem_manager_type = default_memory_manager;
        const static long NR = 0;
        const static long NC = 0;
        const static long cost = 1;
    };
}
    
namespace dlibx {
    /** \brief Matrix of type T where each row is aligned to cache_line_bytes.
     */
    template <typename T, unsigned cache_line_bytes>
    class aligned_matrix
        : public dlib::matrix_exp<aligned_matrix<T,cache_line_bytes> > {

        using matrix_exp =
            dlib::matrix_exp<aligned_matrix<T,cache_line_bytes> >;
        
        stdx::aligned_ptr<T[]> m_buffer;
        std::size_t m_els_allocated = 0;
        long m_nr = 0, m_nc = 0, m_els_per_row = 0;
        
    public:
        using type = T;
        using const_ret_type = const T&;

        aligned_matrix() = default;

        aligned_matrix(aligned_matrix&& other)
            : m_buffer(move(other.m_buffer)),
              m_els_allocated(other.m_els_allocated),
              m_nr(other.m_nr),
              m_nc(other.m_nc),
              m_els_per_row(other.m_els_per_row) {
            other.m_els_allocated = 0;
            other.m_nr = other.m_nc = 0;
        }

        aligned_matrix& operator=(aligned_matrix&& other) {
            m_buffer = move(other.m_buffer);
            m_els_allocated = other.m_els_allocated;
            m_nr = other.m_nr;
            m_nc = other.m_nc;
            m_els_per_row = other.m_els_per_row;
            other.m_els_allocated = 0;
            other.m_nr = other.m_nc = 0;
            return *this;
        }

        /** \brief Set matrix size.
         *
         * Does nothing if size is not changed.
         */
        void set_size(long rows, long cols) {
            if (rows <= 0 || cols <= 0) {
                m_nr = m_nc = 0;
                return;
            }
            if (rows == m_nr && cols == m_nc)
                return;  // do nothing
            static constexpr auto per_block = cache_line_bytes / sizeof(T);
            const auto per_row = 1 + (std::size_t(cols-1) | (per_block-1));
            const auto els_needed = std::size_t(rows) * per_row;
            if (m_els_allocated < els_needed) {
                assert(((els_needed*sizeof(T))&(cache_line_bytes-1)) == 0);
                m_buffer = stdx::make_aligned<T[],cache_line_bytes>(els_needed);
                assert((std::size_t(m_buffer.get())&(cache_line_bytes-1)) == 0);
                m_els_allocated = els_needed;
            }
            m_els_per_row = long(per_row);
            m_nr = rows;
            m_nc = cols;
        }

        aligned_matrix(long rows, long cols) {
            set_size(rows, cols);
        }

        template <typename U>
        bool aliases(const dlib::matrix_exp<U>&) const {
            return false;
        }
        bool aliases(const matrix_exp& item) const {
            return this == &item;
        }
        template <typename U>
        bool destructively_aliases(const dlib::matrix_exp<U>&) const {
            return false;
        }

        inline bool empty() const { return m_nr == 0; }
        inline long nr() const { return m_nr; }
        inline long nc() const { return m_nc; }
        inline long elements_per_row() const { return m_els_per_row; }

        inline type& operator()(long r, long c) {
            return m_buffer[std::size_t(r*m_els_per_row + c)];
        }
        inline const_ret_type operator()(long r, long c) const {
            return m_buffer[std::size_t(r*m_els_per_row + c)];
        }

        aligned_matrix& operator=(const aligned_matrix& other) {
            if (this != &other) {
                set_size(other.nr(), other.nc());
                assert(m_els_per_row == other.m_els_per_row);
                std::copy_n(
                    other.m_buffer.get(), m_nr*m_els_per_row, m_buffer.get());
            }
            return *this;
        }

        aligned_matrix(const aligned_matrix& other) : matrix_exp() {
            *this = other;
        }
    };
}
