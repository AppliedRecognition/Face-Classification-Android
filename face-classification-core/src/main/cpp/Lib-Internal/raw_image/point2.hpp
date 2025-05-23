#pragma once

#include <type_traits>
#include <array>
#include <cmath>

namespace raw_image {
    /** \brief Point or vector in 2d.
     */
    template <typename T>
    struct point2 {
        static_assert(std::is_signed_v<T>);
        using value_type = T;
        T x, y;
    };

    // negation
    template <typename T>
    constexpr auto
    operator-(point2<T> a) {
        a.x = -a.x, a.y = -a.y;
        return a;
    }

    // addition
    template <typename T, typename U>
    constexpr auto&
    operator+=(point2<T>& a, const point2<U>& b) {
        static_assert(std::is_floating_point_v<T> ||
                      !std::is_floating_point_v<U>);
        static_assert(sizeof(T) >= sizeof(U));
        using V = std::conditional_t<std::is_floating_point_v<T>, T, U>;
        return a.x += V(b.x), a.y += V(b.y), a;
    }
    template <typename T>
    constexpr auto
    operator+(point2<T> a, const point2<T>& b) {
        return a += b, a;
    }

    // subtraction
    template <typename T, typename U>
    constexpr auto&
    operator-=(point2<T>& a, const point2<U>& b) {
        static_assert(std::is_floating_point_v<T> ||
                      !std::is_floating_point_v<U>);
        static_assert(sizeof(T) >= sizeof(U));
        using V = std::conditional_t<std::is_floating_point_v<T>, T, U>;
        return a.x -= V(b.x), a.y -= V(b.y), a;
    }
    template <typename T>
    constexpr auto
    operator-(point2<T> a, const point2<T>& b) {
        return a -= b, a;
    }

    // multiplication
    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, point2<T>&>
    operator*=(point2<T>& a, U b) {
        static_assert(std::is_floating_point_v<T> ||
                      !std::is_floating_point_v<U>);
        static_assert(std::is_signed_v<U>);
        using V = std::conditional_t<std::is_floating_point_v<T>, T, U>;
        return a.x *= V(b), a.y *= V(b), a;
    }
    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, point2<T> >
    operator*(point2<T> a, U b) {
        return a *= b, a;
    }
    template <typename U, typename T>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, point2<T> >
    operator*(U b, point2<T> a) {
        return a *= b, a;
    }

    // division
    template <typename T>
    constexpr auto& operator/=(point2<T>& a, T b) {
        return a.x /= b, a.y /= b, a;
    }

    // dot product
    template <typename R, typename T, typename U>
    constexpr auto dot(const point2<T>& a, const point2<U>& b) {
        return R(a.x) * R(b.x) + R(a.y) * R(b.y);
    }
    template <typename T, typename U>
    constexpr auto dot(const point2<T>& a, const point2<U>& b) {
        return dot<decltype(a.x*b.x),T,U>(a,b);
    }
    template <typename R, typename T>
    constexpr auto length_squared(const point2<T>& a) {
        return dot<R,T,T>(a,a);
    }
    template <typename T>
    constexpr auto length_squared(const point2<T>& a) {
        return dot(a,a);
    }

    // z coordinate of cross product for vectors in XY plane (with z == 0)
    template <typename R, typename T, typename U>
    constexpr auto cross(const point2<T>& a, const point2<U>& b) {
        return R(a.x) * R(b.y) - R(a.y) * R(b.x);
    }
    template <typename T, typename U>
    constexpr auto cross(const point2<T>& a, const point2<U>& b) {
        return cross<decltype(a.x*b.x),T,U>(a,b);
    }

    // equality (only for integer values)
    template <typename T, typename U>
    constexpr std::enable_if_t<!std::is_floating_point_v<T> && !std::is_floating_point_v<U>, bool>
    operator==(const point2<T>& a, const point2<U>& b) {
        return a.x == b.x && a.y == b.y;
    }
    template <typename T, typename U>
    constexpr std::enable_if_t<!std::is_floating_point_v<T> && !std::is_floating_point_v<U>, bool>
    operator!=(const point2<T>& a, const point2<U>& b) {
        return a.x != b.x || a.y != b.y;
    }

    using point2i = point2<int>;
    using point2l = point2<long>;
    using point2f = point2<float>;
    using point2d = point2<double>;


    /** \brief Rotated bounding box.
     */
    struct rotated_box {
        point2f center;
        float width, height;
        float angle; // radians
    };

    /** \brief Compute corners of rotated box.
     *
     * Order is clockwise top-left, top-right, bottom-right, bottom-left.
     */
    inline std::array<point2f,4> corners(const rotated_box& rbox) {
        auto right = point2f{std::cos(rbox.angle), std::sin(rbox.angle)};
        auto down  = point2f{-right.y,right.x};
        right *= rbox.width/2;
        down  *= rbox.height/2;
        return {
            rbox.center - right - down,
            rbox.center + right - down,
            rbox.center + right + down,
            rbox.center - right + down
        };
    }
}
