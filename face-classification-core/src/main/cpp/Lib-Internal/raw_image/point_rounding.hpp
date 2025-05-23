#pragma once

#include <stdext/rounding.hpp>

namespace raw_image {

    /** \brief Identify point like class having some kind of x and y.
     */
    template <typename PT, typename Enable = void>
    struct has_xy_var : std::integral_constant<bool,false> {
    };
    template <typename PT>
    struct has_xy_var<PT, std::enable_if_t<
        !std::is_void<decltype(std::declval<PT>().x)>::value &&
        !std::is_void<decltype(std::declval<PT>().y)>::value> >
        : std::integral_constant<bool,true> {
    };
    template <typename PT, typename Enable = void>
    struct has_xy_member : std::integral_constant<bool,false> {
    };
    template <typename PT>
    struct has_xy_member<PT, std::enable_if_t<
        !std::is_void<decltype(std::declval<PT>().x())>::value &&
        !std::is_void<decltype(std::declval<PT>().y())>::value> >
        : std::integral_constant<bool,true> {
    };
    template <typename PT>
    using has_xy = std::integral_constant<
        bool, has_xy_var<PT>::value || has_xy_member<PT>::value>;


    /** \brief Wrapper for point like class having some kind of x and y.
     *
     * Provides access to x and y, and will convert to other such classes
     * (rounding if necessary).
     */
    template <typename PT, typename Enable = void>
    struct xy_wrapper;

    template <typename PT>
    struct xy_wrapper<PT, std::enable_if_t<has_xy_var<PT>::value> > {
        using x_type = std::decay_t<decltype(std::declval<PT>().x)>;
        using y_type = std::decay_t<decltype(std::declval<PT>().y)>;

        using x_reference =
            std::conditional_t<std::is_const_v<PT>, const x_type&, x_type&>;
        using y_reference =
            std::conditional_t<std::is_const_v<PT>, const y_type&, y_type&>;

        const PT p;

        inline x_reference x() { return p.x; }
        inline y_reference y() { return p.y; }

        constexpr inline x_type x() const { return p.x; }
        constexpr inline y_type y() const { return p.y; }
        
        template <typename U, typename = std::enable_if_t<has_xy<U>::value> >
        inline operator U() const {
            return { stdx::round_from(x()), stdx::round_from(y()) };
        }
    };

    template <typename PT>
    struct xy_wrapper<PT, std::enable_if_t<has_xy_member<PT>::value> > {
        using x_type = std::decay_t<decltype(std::declval<PT>().x())>;
        using y_type = std::decay_t<decltype(std::declval<PT>().y())>;

        using x_reference =
            std::conditional_t<std::is_const_v<PT>, const x_type&, x_type&>;
        using y_reference =
            std::conditional_t<std::is_const_v<PT>, const y_type&, y_type&>;

        const PT p;

        inline x_reference x() { return p.x(); }
        inline y_reference y() { return p.y(); }

        constexpr inline x_type x() const { return p.x(); }
        constexpr inline y_type y() const { return p.y(); }
        
        template <typename U, typename = std::enable_if_t<has_xy<U>::value> >
        inline operator U() const {
            return { stdx::round_from(x()), stdx::round_from(y()) };
        }
    };

    /** \brief Helper method to construct xy_wrapper.
     */
    template <typename PT>
    constexpr inline std::enable_if_t<has_xy<PT>::value, xy_wrapper<PT> >
    round_from(const PT& p) {
        return { p };
    }

    /** \brief Explicit point conversion.
     */
    template <typename DEST, typename SRC>
    constexpr inline std::enable_if_t<has_xy<SRC>::value, DEST>
    round_to(const SRC& p) {
        return round_from(p);
    }
}
