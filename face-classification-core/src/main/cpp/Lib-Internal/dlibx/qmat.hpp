#pragma once

#include <dlib/matrix/matrix.h>
#include "aligned_matrix.hpp"
#include "img2col.hpp"

namespace dlibx {
    
    /** \brief Float32 matrix quantized to int16 or int8.
     *
     * Quantization is done per row in the sense that each row has
     * a distinct floating point coefficient.
     * To convert back to float32, multiply each integer matrix element by
     * the corresponding per-row coefficient.
     * That is: coeff(row)*value(row,col).
     *
     * The purpose of this quantization is to make the matrix multiplication
     * LHS * transpose(RHS) as fast as possible.
     * Further, the LHS is expected to be a fixed static matrix which
     * may be quantized by a slower (accuracy preserving) method.
     * The RHS is dynamic data so its quantization has to be fast.
     * To this end, quantization of the LHS returns a value called rhs_limit
     * which is required for correct quantization of the RHS.
     *
     * The matrix multiplication is performed by multiplying pairs of integers
     * to get a sequence of int32 which are then summed.
     * Quantization was performed so as to ensure the int32 will not overflow.
     * The final step is to multiply the int32 by the per-row coefficients
     * to give a float32 result.
     *
     * The floating point coefficient is actually in bfloat16 format.
     * That is, the least significant 16 bits are zero.
     */
    class qmat {
    protected:
        /* Represenation:
         *  data holds both integer matrix elements and per-row float coeffs
         *  nrows and ncols is the effective size of matrix
         *  data.nr() > nrows (one or more extra rows)
         *  the extra rows are allocated for holding the row coefficients
         */
        dlibx::aligned_matrix<float,64> data;
        long nrows = 0, ncols = 0;
        float* row_coeff = nullptr;
        int m_rhs_limit = 0;
        unsigned bytes_per_value;

        explicit qmat(unsigned bytes_per_value)
            : bytes_per_value(bytes_per_value) {}

        qmat(qmat&&) = default;
        qmat& operator=(qmat&&) = default;

        qmat(const qmat&);
        qmat& operator=(const qmat&);

        virtual int calc_rhs_limit() const = 0;


    public:
        virtual ~qmat() = default;

        inline long nr() const { return nrows; }
        inline long nc() const { return ncols; }

        inline bool empty() const {
            return nr() <= 0 || nc() <= 0;
        }
        inline std::size_t size() const {
            return empty() ? 0 : std::size_t(nr()) * std::size_t(nc());
        }

        void set_size(long rows, long cols);

        inline const float& coeff(long row) const {
            return *(row_coeff+row);
        }
        inline float& coeff(long row) {
            return *(row_coeff+row);
        }

        /** \brief Deserialize.
         */
        static std::shared_ptr<qmat> deserialize_shared(std::istream& in);

        /** \brief Serialize.
         */
        virtual void serialize(std::ostream& out) const = 0;
        friend void serialize(const qmat& item, std::ostream& out) {
            item.serialize(out);
        }

        /** \brief Number of bits per element needed for serialize.
         */
        virtual unsigned serialize_bits() const = 0;

        /** \brief Limit to provide when doing assign_rhs() or img2col().
         *
         * If this matrix was not setup as a LHS using assign_lhs() or
         * deserialize(), then zero is returned.
         */
        inline int rhs_limit() const {
            return m_rhs_limit;
        }
        inline int reduce_rhs_limit(int limit) {
            return m_rhs_limit = std::min(m_rhs_limit, limit);
        }
        inline int reset_rhs_limit(int limit) {
            return m_rhs_limit = std::min(calc_rhs_limit(), limit);
        }

        /** \brief Fully connected matrix multiply.
         *
         * Effectively: mult_transpose_rhs(assign_rhs(...))
         * but with output transposed.
         *
         * The input (rhs) is considered to be input.num_samples() rows
         * with input.k()*nr()*nc() columns each.
         *
         * The output is transposed with
         * output.num_samples() == input.num_samples() and
         * output.k()*nr()*nc() == this->nr().
         */
        virtual void fc(const dlib::tensor& input,
                        dlib::resizable_tensor& output) const = 0;

        /** \brief Convolution 1x1.
         *
         * Effectively: mult_transpose_rhs(img2col(...)).
         *
         * \post output.k() == nr()
         */
        virtual void conv1x1(const dlib::tensor& input,
                             dlib::resizable_tensor& output) const = 0;

        /** \brief Convolution.
         *
         * Effectively: mult_transpose_rhs(img2col(...)).
         *
         * The 'd' args are for dilution.
         * The 's' args are for stride.
         *
         * \pre all args >= 1
         *
         * \post output.k() == nr()
         */
        virtual void conv(const dlib::tensor& input,
                          dlib::resizable_tensor& output,
                          int nr, int nc, int dy, int dx,
                          int sy, int sx) const = 0;

        /** \brief Depth-wise convolution.
         *
         * \pre nr > 1 || nc > 1
         * \pre all args >= 1
         * \pre  nr() == mult * input.k() for some mult >= 1
         *
         * \post output.k() == nr()
         */
        virtual void convdw(const dlib::tensor& input,
                            dlib::resizable_tensor& output,
                            int nr, int nc, int dy, int dx,
                            int sy, int sx) const = 0;
    };


    /** \brief Specialization for specific integer size.
     */
    template <typename T>
    class qmat_t final : public qmat {
        static_assert(std::is_same_v<T,int8_t> ||
                      std::is_same_v<T,int16_t>);

        int calc_rhs_limit() const override;

        void quantize_row(long r, const float* src, int limit, float vmax);

        template <int s_nr, int s_nc, int s_dy, int s_dx>
        void conv(const dlib::tensor& input, dlib::tensor& output,
                  int sy, int sx,
                  int nr = s_nr, int nc = s_nc,
                  int dy = s_dy, int dx = s_dx) const;

        template <int s_nr, int s_nc, int s_dy, int s_dx>
        void convdw(const dlib::tensor& input, dlib::tensor& output,
                    int sy, int sx,
                    int nr = s_nr, int nc = s_nc,
                    int dy = s_dy, int dx = s_dx) const;

    public:
        using value_type = T;

        qmat_t() : qmat(unsigned(sizeof(T))) {}
        qmat_t(qmat_t&&) = default;
        qmat_t& operator=(qmat_t&&) = default;
        qmat_t(const qmat_t&) = default;
        qmat_t& operator=(const qmat_t&) = default;

        inline auto ptr(long row) const {
            return reinterpret_cast<const value_type*>(&data(row,0));
        }
        inline auto ptr(long row) {
            return reinterpret_cast<value_type*>(&data(row,0));
        }

        inline const value_type& value(long row, long col) const {
            return *(ptr(row) + col);
        }
        inline value_type& value(long row, long col) {
            return *(ptr(row) + col);
        }

        /** \brief Quantize a single row to specified limit.
         */
        friend inline void
        quantize_row(qmat_t& dest, long r, const float* src,
                     int limit, float vmax) {
            dest.quantize_row(r,src,limit,vmax);
        }

        /** \brief Quantize LHS and return limit for RHS.
         */
        int assign_lhs(const dlib::matrix<float>& lhs, int bits);

        /** \brief Quantize RHS to limit determined by LHS.
         */
        void assign_rhs(const dlib::matrix<float>& rhs, int rhs_limit);

        /** \brief Do img2col and quantize RHS to limit determined by LHS.
         */
        void img2col(int rhs_limit, const img2col_base& gen,
                     const dlib::tensor& input, long n);

        /** \brief Matrix multiply LHS (this) by transpose(RHS).
         */
        dlibx::aligned_matrix<float,64>&
        mult_transpose_rhs(
            const qmat_t& rhs, dlibx::aligned_matrix<float,64>& dest) const;

        inline dlibx::aligned_matrix<float,64>
        mult_transpose_rhs(const qmat_t& rhs) const {
            dlibx::aligned_matrix<float,64> result;
            mult_transpose_rhs(rhs, result);
            return result;
        }

        /** \brief Fully connected matrix multiply.
         *
         * Effectively: mult_transpose_rhs(assign_rhs(...))
         * but with output transposed.
         */
        void fc(const dlib::tensor& input,
                dlib::resizable_tensor& output) const override;

        /// pointwise convolution
        void conv1x1(const dlib::tensor& input,
                     dlib::resizable_tensor& output) const override;

        /// general convolution
        void conv(const dlib::tensor& input,
                  dlib::resizable_tensor& output,
                  int nr, int nc, int dy, int dx,
                  int sy, int sx) const override;

        /// depthwise convolution
        void convdw(const dlib::tensor& input,
                    dlib::resizable_tensor& output,
                    int nr, int nc, int dy, int dx,
                    int sy, int sx) const override;


        /** \brief Serialize.
         */
        void serialize(std::ostream& out) const override;
        unsigned serialize_bits() const override;

        /** \brief Deserialize.
         */
        void deserialize(std::istream& in, unsigned bits);
        void deserialize_1(std::istream& in);
        void deserialize_2(std::istream& in);
    };

    using qmat8 = qmat_t<int8_t>;
    using qmat16 = qmat_t<int16_t>;
}
