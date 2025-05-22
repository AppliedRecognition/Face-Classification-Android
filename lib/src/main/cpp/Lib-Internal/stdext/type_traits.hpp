#pragma once

#include <type_traits>

namespace stdx {
    /** \brief Is first type same as any of the others.
     */
    template <typename T, typename...>
    struct is_one_of : std::false_type {};
    template <typename T, typename T0, typename... Ts>
    struct is_one_of<T,T0,Ts...>
        : std::conditional<std::is_same<T,T0>::value,
                           std::true_type,
                           is_one_of<T,Ts...> >::type {};

    
    /** \brief Is bool or decays to bool.
     */
    template <typename T>
    using is_bool = std::is_same<typename std::decay<T>::type, bool>;


    /** \brief Same as is_integral<> except false for bool and char types.
     *
     * Rational: neither 'bool' nor 'char' are truely integer types.  
     * Note that for a tiny integer one should use either 'signed char' or
     * 'unsigned char'.
     */
    template <typename T>
    using is_pure_integral = std::integral_constant<
        bool,
        std::is_integral<T>::value &&
        !is_one_of<typename std::decay<T>::type,
                   bool,char,char16_t,char32_t,wchar_t>::value>;

    
    /** \brief Same as is_arithmetic<> except false for bool and char types.
     *
     * Rational: neither 'bool' nor 'char' are truely integer types.  
     * Note that for a tiny integer one should use either 'signed char' or
     * 'unsigned char'.
     */
    template <typename T>
    using is_pure_arithmetic = std::integral_constant<
        bool,
        std::is_arithmetic<T>::value &&
        !is_one_of<typename std::decay<T>::type,
                   bool,char,char16_t,char32_t,wchar_t>::value>;


    /** \brief Test for range as defined for range-based for loop.
     *
     * std::begin() and std::end() must be defined for range and
     * iterator must dereference to non-void
     */
    template <typename T, typename = void>
    struct is_range : std::false_type {};
    template <typename T>
    struct is_range<
        T, typename std::enable_if<
               std::is_same<typename std::decay<decltype(std::begin(std::declval<T>()))>::type,
                            typename std::decay<decltype(std::end(std::declval<T>()))>::type>::value &&
               !std::is_void<decltype(*std::begin(std::declval<T>()))>::value>::type>
        : std::true_type {};
    
    /// decltype(*std::begin()) -- undefined for non-range
    template <typename T>
    using range_value_type = typename std::remove_reference<
        decltype(*std::begin(std::declval<T>()))>::type;

    /** \brief Depth of nested range.
     *
     * For non-range, depth is 0.
     * For infinite depth range, depth is range_depth_limit.
     */
    static constexpr auto range_depth_limit = 16;
    template <typename T, unsigned N = 0, typename = void>
    struct range_depth : std::integral_constant<unsigned, N> {};
    template <typename T, unsigned N>
    struct range_depth<
        T, N, typename std::enable_if<
                  (N<range_depth_limit) && is_range<T>::value>::type>
        : range_depth<range_value_type<T>, N+1> {};

    /** \brief Test range for 0 < depth < limit.
     */
    template <typename T>
    using is_range_depth_finite = std::integral_constant<
        bool,
        is_range<T>::value && (range_depth<T>::value < range_depth_limit)>;
}
