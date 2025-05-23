#pragma once

#include <cstdint>
#include <vector>
#include <iterator>
#include <istream>
#include <stdext/binary.hpp>
#include <stdext/forward_iterator.hpp>
#include <stdext/aligned_alloc.hpp>


/** \file fpvc.hpp
 * \brief Floating point vector compression.
 *
 * Methods to encode a zero-centered vector of float to 1-byte per value.
 */

namespace rec {
    namespace internal {

        /** \brief Compress unsigned integer.
         *
         * Zero and one map to themselves.
         * Larger input values map to smaller output values.
         * Rounding is used to minimize the difference between the input
         * value and an inverse mapping of the output value.
         * Negative input values result in zero output value.
         *
         * As an example, the integers [0,1708] map to [0,127].
         */
        unsigned fpvc_unsigned_compress(int x);

        /** \brief The inverse of fpvc_unsigned_compress().
         */
        int fpvc_unsigned_decompress(unsigned x);


        /** \brief Compressed vector representation.
         *
         * To recover a specific element, use either of the decompress tables
         * above.
         * Element i can be calculated as pair.first * table[pair.second[i]].
         */
        using fpvc_vector_type = std::pair<float,std::vector<uint8_t> >;


        /** \brief Tables for decoding 8-bit fpvc encoding to short or float.
         *
         * The range of the output value is [-1984,1984].
         * As a short, both encodings of zero map to zero, while
         * as a float, they map to 0 and -0.
         */
        extern const int16_t fpvc_s16_decompress_table[256];
        extern const float fpvc_f32_decompress_table[256];


        /** \brief Decompress iterator adaptor.
         */
        template <typename ITER>
        struct fpvc_s16_decompress_iterator_adaptor {
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = int16_t;
            using pointer = const int16_t*;
            using reference = const int16_t&;

            ITER iter;
            explicit fpvc_s16_decompress_iterator_adaptor(ITER iter)
                : iter(iter) {}

            inline bool operator==(
                const fpvc_s16_decompress_iterator_adaptor& other) const {
                return iter == other.iter;
            }
            inline bool operator!=(
                const fpvc_s16_decompress_iterator_adaptor& other) const {
                return iter != other.iter;
            }

            inline fpvc_s16_decompress_iterator_adaptor& operator++() {
                ++iter;
                return *this;
            }
            inline fpvc_s16_decompress_iterator_adaptor operator++(int) {
                auto other = *this;
                ++iter;
                return other;
            }
            template <typename I>
            inline fpvc_s16_decompress_iterator_adaptor operator+(I n) const {
                auto other = *this;
                other.iter += n;
                return other;
            }

            inline int16_t operator*() const {
                return fpvc_s16_decompress_table[*iter];
            }
        };
        template <typename ITER>
        inline fpvc_s16_decompress_iterator_adaptor<ITER>
        fpvc_s16_decompress_iterator(ITER iter) {
            return fpvc_s16_decompress_iterator_adaptor<ITER>(iter);
        }


        /** \brief Decompress iterator adaptor.
         */
        template <typename ITER>
        struct fpvc_f32_decompress_iterator_adaptor {
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = float;
            using pointer = const float*;
            using reference = const float&;

            ITER iter;
            explicit fpvc_f32_decompress_iterator_adaptor(ITER iter)
                : iter(iter) {}

            inline bool operator==(
                const fpvc_f32_decompress_iterator_adaptor& other) const {
                return iter == other.iter;
            }
            inline bool operator!=(
                const fpvc_f32_decompress_iterator_adaptor& other) const {
                return iter != other.iter;
            }

            inline fpvc_f32_decompress_iterator_adaptor& operator++() {
                ++iter;
                return *this;
            }
            inline fpvc_f32_decompress_iterator_adaptor operator++(int) {
                auto other = *this;
                ++iter;
                return other;
            }
            inline fpvc_f32_decompress_iterator_adaptor
            operator+(typename ITER::difference_type n) const {
                auto other = *this;
                other.iter += n;
                return other;
            }

            inline float operator*() const {
                return fpvc_f32_decompress_table[*iter];
            }
        };
        template <typename ITER>
        inline fpvc_f32_decompress_iterator_adaptor<ITER>
        fpvc_f32_decompress_iterator(ITER iter) {
            return fpvc_f32_decompress_iterator_adaptor<ITER>(iter);
        }


        /** \brief Inverse of fpvc_vector_compress().
         */
        template <typename ITER>
        void fpvc_vector_decompress(ITER out, const fpvc_vector_type& enc) {
            for (const auto y : enc.second) {
                *out = enc.first * fpvc_f32_decompress_table[y];
                ++out;
            }
        }

        /** \brief Inverse of fpvc_vector_compress().
         */
        std::vector<float> fpvc_vector_decompress(const fpvc_vector_type& enc);


        /** \brief Compress a vector of float values to 1-byte per value.
         *
         * The input vector is assumed to be zero-centered.
         * That is, the mean of the values should be near or equal to zero
         * and the values should be roughly normally distributed about zero.
         * Specicially, the variance to the negative side should be about the
         * same as the variance to the positive side.
         *
         * The compression preserves the maximum absolute value.
         * The element that assumes the maximum absolute value will be
         * restored exactly by the decompress function.
         * All other elements may lose precision.
         * Very small elements (in absolute value) may be quantized to zero
         * (negative zero for negative values).
         *
         * Also, an iterative process is used to try to preserve the norm
         * of the vector (unless no_opt is true).
         * Note that the result of this optimization is not exact.
         *
         * If no_opt is true, then the compression is faster but there will
         * be more error in the norm of the decompressed vector.
         *
         * The returned vector will have space allocated to a multiple of
         * 4 bytes.  This makes padding to a multiple of 4 bytes easier.
         */
        fpvc_vector_type fpvc_vector_compress(
            stdx::forward_iterator<float> first,
            stdx::forward_iterator<float> last,
            bool no_opt = false);


        /** \brief Return required space (in bytes) needed for serialization.
         *
         * The result is 4 + vector_size + padding to multiple of 4-bytes.
         * The result will be a multiple of 4 bytes.
         */
        inline std::size_t fpvc_vector_serialize_size(std::size_t n) {
            // 7 = 4-byte vec.first + vec.second.size() rounded up to 4-byte
            return (n+7) & ~std::size_t(3);
        }
        inline std::size_t
        fpvc_vector_serialize_size(const fpvc_vector_type& vec) {
            return fpvc_vector_serialize_size(vec.second.size());
        }

        /** \brief Serialize fpvc vector.
         */
        void fpvc_vector_serialize(std::vector<unsigned char>& dest,
                                   const fpvc_vector_type& vec);

        /** \brief Deserialize fpvc vector from memory.
         *
         * Note that vector_size is the size of the resulting vector,
         * not the number of bytes available at src.
         * Use fpvc_vector_serialize_size() to determine how many bytes are
         * consumed from src.
         */
        fpvc_vector_type fpvc_vector_deserialize(
            const void* src, std::size_t vector_size);

        /** \brief Deserialize fpvc vector from stream.
         *
         * Note that vector_size is the size of the resulting vector,
         * not the number of bytes available at src.
         * Use fpvc_vector_serialize_size() to determine how many bytes are
         * consumed from src.
         */
        fpvc_vector_type fpvc_vector_deserialize(
            std::istream& in, std::size_t vector_size);


        /** \brief Fixed point vector with 16-bit signed elements.
         *
         * Semantically the value of element i is coeff * values[i].
         *
         * The values vector is aligned to 32-bytes so as to work with
         * AVX 256-bit operations (and SSE and NEON too).
         */
        struct fp16vec {
            stdx::aligned_ptr<int16_t[]> values;
            std::size_t m_size = 0;
            float coeff = 0;

            inline void resize(std::size_t n) {
                if (m_size < n)
                    values = stdx::make_aligned<int16_t[],32>(n);
                m_size = n;
            }

            constexpr auto empty() const { return m_size == 0; }
            constexpr auto size() const { return m_size; }
            auto* begin() { return values.get(); }
            auto const* begin() const { return values.get(); }
            auto const* cbegin() const { return values.get(); }
            auto* end() { return begin() + m_size; }
            auto const* end() const { return begin() + m_size; }
            auto const* cend() const { return begin() + m_size; }

            fp16vec() = default;
            fp16vec(fp16vec&&) = default;
            fp16vec& operator=(fp16vec&&) = default;
            fp16vec(const fp16vec& other)
                : coeff(other.coeff) {
                resize(other.size());
                std::copy(other.begin(), other.end(), begin());
            }
        };

        fp16vec to_fp16vec(const fpvc_vector_type& vec);

        /** \brief Size in bytes of 12-bit serialization with padding.
         *
         * Includes 4-byte float coefficient but not vector size integer.
         * Padding is to a multiple of 4 bytes.
         */
        inline std::size_t fp16vec_12_bytes(std::size_t n) {
            return 4 * (1 + (3*n+7)/8);
        }

        /** \brief Size in bytes of 16-bit serialization with padding.
         *
         * Includes 4-byte float coefficient but not vector size integer.
         * Padding is to a multiple of 4 bytes.
         */
        inline std::size_t fp16vec_16_bytes(std::size_t n) {
            return 2*(2+n+(n&1));
        }

        /** \brief Number of bits required to serialize without loss.
         */
        unsigned bits_required(const fp16vec& vec);

        /** \brief Serialize to 8-bit format.
         *
         * This method will scale values if any value is outside [-127,127].
         *
         * The format is 2 byte bfloat16 coefficient followed by the
         * bytes of the vector.  No padded is done.
         */
        void serialize_8(std::vector<unsigned char>& dest, const fp16vec& vec);

        /** \brief Serialize to 12-bit format.
         *
         * An exception is thrown if any value is outside [-2048, 2047].
         */
        void serialize_12(std::vector<unsigned char>& dest, const fp16vec& vec);

        /** \brief Serialize to 16-bit format.
         */
        void serialize_16(std::vector<unsigned char>& dest, const fp16vec& vec);

        /** \brief Deserialize 12-bit format.
         *
         * Note that vector_size is the size of the resulting vector,
         * not the number of bytes available at src.
         */
        fp16vec deserialize_fp16vec_12(
            const void* src, std::size_t vector_size);

        /** \brief Deserialize 16-bit format.
         *
         * Note that vector_size is the size of the resulting vector,
         * not the number of bytes available at src.
         */
        fp16vec deserialize_fp16vec_16(
            const void* src, std::size_t vector_size);



        /// prototype serialization
        stdx::binary serialize(
            unsigned version,
            stdx::forward_iterator<const fpvc_vector_type&> first,
            stdx::forward_iterator<const fpvc_vector_type&> last);
        stdx::binary serialize(
            unsigned version,
            stdx::forward_iterator<const fp16vec&> first,
            stdx::forward_iterator<const fp16vec&> last,
            unsigned bit_per_element = 12);

        /** \brief Deserialize prototype.
         *
         * The second (fpvc_vector_type) will only be non-empty
         * if the data was in 8-bit format.
         */
        std::vector<std::pair<fp16vec, fpvc_vector_type> >
        deserialize_fpvc(const void* src, std::size_t len);
    }
}
