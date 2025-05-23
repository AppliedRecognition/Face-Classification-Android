#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

namespace stdx {

    // std::round() is not available on android
    inline float round(float x) { return roundf(x); }
    inline double round(double x) { return ::round(x); }
    inline long double round(long double x) { return roundl(x); }

    template <typename T>
    inline std::enable_if_t<std::is_integral_v<T>, double> round(T x) {
        return ::round(double(x));
    }
    
    /** \brief Round to non-integral (floating point).
     *
     * If the floating point type cannot exactly represent the argument,
     * then the return value is rounded to a value that can be
     * represented.
     */
    template <typename To, typename From>
    constexpr inline
    std::enable_if_t<!std::is_integral_v<To> && std::is_arithmetic_v<From>, To>
    round_to(From x) {
        return To(x);
    }

    /** \brief Round to integral from non-integral (floating point).
     *
     * On overflow, the positive or negative limit of the return type
     * is returned.
     */
    template <typename To, typename From>
    inline
    std::enable_if_t<std::is_integral_v<To> && !std::is_integral_v<From>, To>
    round_to(From x) {
        using LIM = std::numeric_limits<To>;
        static constexpr auto half = 1 + LIM::max()/2;
        static_assert((half&(half-1)) == 0,
                      "half maximum integer must be a power of 2");
        // note: 2 * half == 1 + LIM::max()
        const auto r = round(x);
        return r > LIM::min() ? 
            (r < 2*From(half) ? To(r) : LIM::max()) : LIM::min();
    }

    /** \brief Clamp integer to positive or negative limit if necessary.
     */
    template <typename To, typename From>
    constexpr inline
    std::enable_if_t<std::is_integral_v<To> &&
                     std::is_integral_v<From> &&
                     !std::is_unsigned_v<From>, To>
    round_to(From x) {  // x is signed
        using S = std::make_signed_t<To>;
        using U = std::conditional_t<
            std::is_unsigned_v<To>, std::make_unsigned_t<From>, From>;
        const auto ToMin = std::numeric_limits<To>::min();
        const auto ToMax = std::numeric_limits<To>::max();
        return x >= S(ToMin) ? (U(x) <= ToMax ? To(x) : ToMax) : ToMin;
    }
    template <typename To, typename From>
    constexpr inline
    std::enable_if_t<std::is_integral_v<To> && std::is_unsigned_v<From>, To>
    round_to(From x) {  // x is unsigned
        const auto ToMax = std::numeric_limits<To>::max();
        return x <= ToMax ? To(x) : ToMax;
    }

    /** \brief Wrapper for value to implicitly round.
     */
    template <typename V>
    struct round_value {
        V v;
        constexpr explicit round_value(V v) : v(v) {}
        template <typename U>
        constexpr inline operator U() const {
            return round_to<U>(v);
        }
    };
    /** \brief Wrap value which will implicitly round as needed.
     */
    template <typename V>
    constexpr inline round_value<V> round_from(V v) {
        return round_value<V>(v);
    }
}
