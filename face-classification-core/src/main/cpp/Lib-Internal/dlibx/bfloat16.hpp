#pragma once

#include <stdext/span.hpp>
#include <istream>
#include <ostream>

namespace dlib {
    class tensor;
    class resizable_tensor;
}

namespace dlibx {
    /** \brief Serialization format for supporting dnn layers.
     *
     * For newly created layers, native is the same as float32.
     * Otherwise, native may serialize to the same format
     * the layer was deserialized from.
     *
     * Note that float32 produces output that stock dlib can read.
     */
    enum class parameter_format {
        native = 0, float32 = 1, bfloat16 = 2, quantize_base = 4
    };
    using pf = parameter_format;

    /** \brief Specific quantize level.
     *
     * The valid range for bits is 4 to 16, inclusive.
     */
    template <typename T>
    constexpr inline auto quantize(T bits) {
        return parameter_format(
            int(pf::quantize_base) + 16 -
            int(bits < 4 ? 4 : bits <= 16 ? bits : 16));
    }
    constexpr inline auto bits_per_element(parameter_format format) {
        if (format == pf::float32)
            return 32;
        if (format == pf::bfloat16)
            return 16;
        const auto bits = 16 - (int(format) - int(pf::quantize_base));
        return (4 <= bits && bits <= 16) ? bits : 0;
    }


    /** \brief Get/set format flag on stream.
     */
    parameter_format get_parameter_format(std::ostream& strm);
    std::ostream& set_parameter_format(std::ostream&, parameter_format);
    inline void serialize(parameter_format pf, std::ostream& out) {
        set_parameter_format(out, pf);
    }


    /** \brief Number of bits required to represent unsigned value.
     *
     * Note that 1 is returned when the input is 0.
     */
    unsigned bits_required(unsigned);

    /** \brief Determine how many bits per element are required to encode range.
     */
    template <typename I>
    std::enable_if_t<std::is_integral_v<I>, unsigned>
    bits_required(I const* data, std::size_t size) {
        unsigned limit = 0;
        for ( ; size > 0; --size, ++data)
            limit = std::max(
                limit, *data < 0 ? unsigned(-(*data+1)) : unsigned(*data));
        return bits_required(std::is_unsigned_v<I> ? limit : (limit<<1));
    }


    /** \brief Wrapper around std::ostream to write integers of a fixed number
     * of bits (not necessarily a multiple of 8).
     *
     * In the case where bits_per_element is a multiple of 8, then the output
     * is essentially little-endian.
     */
    class bits_writer {
        std::ostream& out;
        const unsigned bits_per_element;
        unsigned bits_remaining = 8;
        unsigned char c = 0;

        bits_writer(const bits_writer&) = delete;

    public:
        bits_writer(std::ostream& out, unsigned bits_per_element)
            : out(out), bits_per_element(bits_per_element) {
            if (!(0 < bits_per_element && bits_per_element <= 64))
                throw std::invalid_argument("invalid bits_per_element");
        }

        inline std::ostream& flush() {
            if (bits_remaining < 8) {
                out.write(reinterpret_cast<char const*>(&c),1);
                bits_remaining = 8;
                c = 0;
            }
            return out;
        }
        ~bits_writer() {
            flush();
        }

        explicit operator bool() const { return out.good(); }

        template <typename I>
        std::enable_if_t<std::is_integral_v<I>, std::ostream&>
        operator()(I _x) {
            using T = std::common_type_t<I,int>;
            using U = std::make_unsigned_t<T>;
            auto x = U(T(_x));
            using uchar = unsigned char;
            uchar buf[9] = {0};
            buf[0] = c;
            int len = 0;
            for (auto n = bits_per_element; n > 0; ) {
                const auto mask = ~(~0u << bits_remaining);
                if (n >= bits_remaining) {
                    buf[len] = uchar(unsigned(buf[len]) | (x & mask));
                    x >>= bits_remaining;
                    n -= bits_remaining;
                    bits_remaining = 8;
                    ++len;
                }
                else { // n < bits_remaining
                    x <<= (bits_remaining - n);
                    buf[len] = uchar(unsigned(buf[len]) | (x & mask));
                    bits_remaining -= n;
                    break;
                }
            }
            c = buf[len];
            return out.write(reinterpret_cast<char const*>(buf),len);
        }
    };


    /** \brief Wrapper around std::istream to read integers of a fixed number
     * of bits (not necessarily a multiple of 8).
     */
    class bits_reader {
        std::istream& in;
        const unsigned bits_per_element;
        unsigned bits_remaining = 0;
        unsigned char c = 0;

        bits_reader(const bits_reader&) = delete;

    public:
        bits_reader(std::istream& in, unsigned bits_per_element)
            : in(in), bits_per_element(bits_per_element) {
            if (!(0 < bits_per_element && bits_per_element <= 64))
                throw std::invalid_argument("invalid bits_per_element");
        }

        explicit operator bool() const { return in.good(); }

        template <typename I>
        std::enable_if_t<std::is_integral_v<I>, I> get() {
            unsigned char buf[9];
            if (bits_remaining == 0) {
                in.read(reinterpret_cast<char*>(buf),
                        std::streamsize((bits_per_element + 7) / 8));
                bits_remaining = 8;
            }
            else {
                buf[0] = c;
                if (bits_remaining < bits_per_element)
                    in.read(reinterpret_cast<char*>(buf)+1,
                            std::streamsize((bits_per_element + 7 - bits_remaining) / 8));
            }
            using T = std::common_type_t<
                I, std::conditional_t<std::is_signed_v<I>,int,unsigned> >;
            using U = std::make_unsigned_t<T>;
            U x = 0;
            unsigned char const* p = buf;
            for (auto n = bits_per_element; n > 0; ) {
                if (n > bits_remaining) {
                    x >>= bits_remaining;
                    x |= U(*p) << (8*sizeof(U) - bits_remaining);
                    n -= bits_remaining;
                    ++p;
                    bits_remaining = 8;
                }
                else { // n <= bits_remaining
                    x >>= n;
                    x |= (U(*p) >> (bits_remaining - n)) << (8*sizeof(U) - n);
                    bits_remaining -= n;
                    c = static_cast<unsigned char>(
                        unsigned(*p) & ~(~0u<<bits_remaining));
                    break;
                }
            }
            return I(T(x)>>(8*sizeof(T)-bits_per_element));
        }
    };


    /** \brief Span of const float for serialize.
     *
     * This serialzation format has no headers. 
     * The number of bytes output will be exactly 2 bytes for each float.
     */
    struct bfloat16_const_span : stdx::span<const float> {
        using stdx::span<const float>::span;
        void serialize(std::ostream& out) const;
        friend inline void
        serialize(const bfloat16_const_span& item, std::ostream& out) {
            item.serialize(out);
        }
    };

    /** \brief Span of float for deserialize.
     */
    struct bfloat16_span : stdx::span<float> {
        using stdx::span<float>::span;
        void deserialize(std::istream& in);
        friend inline void
        deserialize(bfloat16_span item, std::istream& in) {
            item.deserialize(in);
        }
        friend inline void
        serialize(const bfloat16_span& item, std::ostream& out) {
            bfloat16_const_span(item).serialize(out);
        }
    };


    /** \brief Method to select bfloat16 serialization.
     *
     * Use the bfloat16 object below as either a method or a format flag.
     */
    struct bfloat16_method {
        /** \brief Construct span for deserialize.
         *
         * Example: deserialize(dlibx::bfloat16(data,size), in);
         */
        template <typename... Args>
        constexpr std::enable_if_t<
            std::is_constructible<bfloat16_span,Args&&...>::value,
            bfloat16_span>
        operator()(Args&&... args) const {
            return bfloat16_span(std::forward<Args>(args)...);
        }

        /** \brief Construct span for serialize.
         *
         * Example: serialize(dlibx::bfloat16(data,size), out);
         */
        template <typename... Args>
        constexpr std::enable_if_t<
            std::is_constructible<bfloat16_const_span,Args&&...>::value &&
            !std::is_constructible<bfloat16_span,Args&&...>::value,
            bfloat16_const_span>
        operator()(Args&&... args) const {
            return bfloat16_const_span(std::forward<Args>(args)...);
        }

        /** \brief Use the bfloat16 object to set serialization format.
         *
         * Example: dlib::serialize(filename) << dlibx::bfloat16 << net_object;
         */
        friend void serialize(const bfloat16_method&, std::ostream& out) {
            set_parameter_format(out, pf::bfloat16);
        }
    };
    const bfloat16_method bfloat16;


    /** \brief Serialize tensor in bfloat16 format.
     */
    void serialize_bfloat16(const dlib::tensor& src, std::ostream& out);


    /** \brief Deserialize tensor.
     *
     * This method will read either float32 or bfloat16 format.
     */
    void deserialize(dlib::resizable_tensor& dest, std::istream& in);


    /** \brief Truncate array of floats to bfloat16 format.
     *
     * \returns data + size
     */
    float* truncate_to_bfloat16(float* data, std::size_t size);


    /** \brief Test if least significant bits are zero for all elements.
     */
    bool is_bfloat16(float const* data, std::size_t size);
    bool is_bfloat16(const dlib::tensor& src);
}
