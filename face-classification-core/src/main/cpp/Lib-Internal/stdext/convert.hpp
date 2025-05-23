#pragma once

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace stdx {

    /** \brief Test if integer of type From can always be converted to type To.
     */
    template <typename To, typename From, typename = void>
    struct is_nothrow_convertible_to : std::false_type {};
    template <typename To, typename From>
    struct is_nothrow_convertible_to<
        To, From,
        std::enable_if_t<(std::is_integral_v<To> &&
                          (std::is_signed_v<To> || std::is_unsigned_v<From>) &&
                          std::numeric_limits<From>::max() <= std::numeric_limits<To>::max())> >
        : std::true_type {};


    /** \brief Convert to non-integral (floating point).
     *
     * If the floating point type cannot exactly represent the argument,
     * then the return value is rounded to a value that can be
     * represented.
     */
    template <typename To, typename From>
    constexpr std::enable_if_t<
        !std::is_integral_v<To> &&
        std::is_arithmetic_v<From>, To>
    convert_to(From x) {
        return To(x);
    }

    /** \brief Convert integer without exception if guaranteed to work.
     */
    template <typename To, typename From>
    constexpr std::enable_if_t<is_nothrow_convertible_to<To,From>::value, To>
    convert_to(From x) {
        return To(x);
    }

    /** \brief Convert integer and throw exception if out of range.
     */
    template <typename To, typename From>
    inline std::enable_if_t<
        std::is_integral_v<To> &&
        std::is_integral_v<From> &&
        !std::is_unsigned_v<From> &&
        !is_nothrow_convertible_to<To,From>::value, To>
    convert_to(From x) {  // x is signed
        using LIM = std::numeric_limits<To>;
        using S = std::make_signed_t<To>;
        using U = std::conditional_t<
            std::is_unsigned_v<To>, std::make_unsigned_t<From>, From>;
        if (S(LIM::min()) <= x && U(x) <= LIM::max())
            return To(x);
        throw std::range_error("signed integer out of range for destination type");
    }

    template <typename To, typename From>
    inline std::enable_if_t<
        std::is_integral_v<To> &&
        std::is_unsigned_v<From> &&
        !is_nothrow_convertible_to<To,From>::value, To>
    convert_to(From x) {  // x is unsigned
        using LIM = std::numeric_limits<To>;
        if (x <= LIM::max())
            return To(x);
        throw std::range_error("unsigned integer too large for destination type");
    }

    /** \brief Wrap value to be implicitly converted when needed.
     */
    template <typename V>
    struct convert_from {
        V v;
        constexpr explicit convert_from(V v) : v(v) {}
        template <typename U>
        constexpr inline operator U() const {
            return convert_to<U>(v);
        }
    };
}
