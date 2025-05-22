#pragma once

#include <raw_image/point2.hpp>

namespace raw_image {
    /** \brief Point or vector in 3d.
     */
    template <typename T>
    struct point3 {
        static_assert(std::is_signed_v<T>);
        using value_type = T;
        T x, y, z;

        constexpr point3() = default;
        constexpr point3(T x, T y, T z) : x(x), y(y), z(z) {}
        constexpr point3(point2<T> xy, T z) : x(xy.x), y(xy.y), z(z) {}
        explicit constexpr point3(point2<T> xy) : x(xy.x), y(xy.y), z(0) {}

        template <typename I>
        constexpr auto& operator[](I i) { return (&x)[i]; }
        template <typename I>
        constexpr T operator[](I i) const { return (&x)[i]; }
    };

    // negation
    template <typename T>
    constexpr auto
    operator-(point3<T> a) {
        a.x = -a.x, a.y = -a.y, a.z = -a.z;
        return a;
    }

    // addition
    template <typename T, typename U>
    constexpr auto&
    operator+=(point3<T>& a, const point3<U>& b) {
        static_assert(std::is_floating_point_v<T> ||
                      !std::is_floating_point_v<U>);
        static_assert(sizeof(T) >= sizeof(U));
        using V = std::conditional_t<std::is_floating_point_v<T>, T, U>;
        return a.x += V(b.x), a.y += V(b.y), a.z += V(b.z), a;
    }
    template <typename T>
    constexpr auto
    operator+(point3<T> a, const point3<T>& b) {
        return a += b, a;
    }

    // subtraction
    template <typename T, typename U>
    constexpr auto&
    operator-=(point3<T>& a, const point3<U>& b) {
        static_assert(std::is_floating_point_v<T> ||
                      !std::is_floating_point_v<U>);
        static_assert(sizeof(T) >= sizeof(U));
        using V = std::conditional_t<std::is_floating_point_v<T>, T, U>;
        return a.x -= V(b.x), a.y -= V(b.y), a.z -= V(b.z), a;
    }
    template <typename T>
    constexpr auto
    operator-(point3<T> a, const point3<T>& b) {
        return a -= b, a;
    }

    // multiplication
    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, point3<T>&>
    operator*=(point3<T>& a, U b) {
        static_assert(std::is_floating_point_v<T> ||
                      !std::is_floating_point_v<U>);
        static_assert(std::is_signed_v<U>);
        using V = std::conditional_t<std::is_floating_point_v<T>, T, U>;
        return a.x *= V(b), a.y *= V(b), a.z *= V(b), a;
    }
    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, point3<T> >
    operator*(point3<T> a, U b) {
        return a *= b, a;
    }
    template <typename U, typename T>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, point3<T> >
    operator*(U b, point3<T> a) {
        return a *= b, a;
    }

    // division
    template <typename T>
    constexpr auto& operator/=(point3<T>& a, T b) {
        return a.x /= b, a.y /= b, a.z /= b, a;
    }

    // dot product
    template <typename R, typename T, typename U>
    constexpr auto dot(const point3<T>& a, const point3<U>& b) {
        return R(a.x) * R(b.x) + R(a.y) * R(b.y) + R(a.z) * R(b.z);
    }
    template <typename T, typename U>
    constexpr auto dot(const point3<T>& a, const point3<U>& b) {
        return dot<decltype(a.x*b.x),T,U>(a,b);
    }
    template <typename R, typename T>
    constexpr auto length_squared(const point3<T>& a) {
        return dot<R,T,T>(a,a);
    }
    template <typename T>
    constexpr auto length_squared(const point3<T>& a) {
        return dot(a,a);
    }

    // cross product
    template <typename R, typename T, typename U>
    constexpr auto cross(const point3<T>& a, const point3<U>& b) {
        return point3<R>{
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }
    template <typename T, typename U>
    constexpr auto cross(const point3<T>& a, const point3<U>& b) {
        return cross<decltype(a.x*b.x),T,U>(a,b);
    }

    // equality (only for integer values)
    template <typename T, typename U>
    constexpr std::enable_if_t<!std::is_floating_point_v<T> && !std::is_floating_point_v<U>, bool>
    operator==(const point3<T>& a, const point3<U>& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
    template <typename T, typename U>
    constexpr std::enable_if_t<!std::is_floating_point_v<T> && !std::is_floating_point_v<U>, bool>
    operator!=(const point3<T>& a, const point3<U>& b) {
        return a.x != b.x || a.y != b.y || a.z != b.z;
    }

    using point3i = point3<int>;
    using point3l = point3<long>;
    using point3f = point3<float>;
    using point3d = point3<double>;

    /** \brief 3x3 matrix.
     */
    template <typename T>
    struct matrix3x3 {
        point3<T> rows[3];
    };

    using matrix3x3i = matrix3x3<int>;
    using matrix3x3l = matrix3x3<long>;
    using matrix3x3f = matrix3x3<float>;
    using matrix3x3d = matrix3x3<double>;

    template <typename T>
    constexpr const auto identity3x3 = matrix3x3<T> {
        point3<T>{1,0,0},
        point3<T>{0,1,0},
        point3<T>{0,0,1},
    };
    constexpr const auto I3x3i = identity3x3<int>;
    constexpr const auto I3x3l = identity3x3<long>;
    constexpr const auto I3x3f = identity3x3<float>;
    constexpr const auto I3x3d = identity3x3<double>;

    // addition
    template <typename T, typename U>
    constexpr auto&
    operator+=(matrix3x3<T>& a, const matrix3x3<U>& b) {
        for (unsigned i = 0; i < 3; ++i)
            a.rows[i] += b.rows[i];
        return a;
    }
    template <typename T>
    constexpr auto operator+(matrix3x3<T> a, const matrix3x3<T>& b) {
        return a += b, a;
    }

    // subtraction
    template <typename T, typename U>
    constexpr auto&
    operator-=(matrix3x3<T>& a, const matrix3x3<U>& b) {
        for (unsigned i = 0; i < 3; ++i)
            a.rows[i] -= b.rows[i];
        return a;
    }
    template <typename T>
    constexpr auto operator-(matrix3x3<T> a, const matrix3x3<T>& b) {
        return a -= b, a;
    }

    // multiplication
    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, matrix3x3<T>&>
    operator*=(matrix3x3<T>& a, U b) {
        for (auto& p : a.rows)
            p *= b;
        return a;
    }
    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, matrix3x3<T> >
    operator*(matrix3x3<T> a, U b) {
        return a *= b, a;
    }
    template <typename U, typename T>
    constexpr std::enable_if_t<std::is_arithmetic_v<U>, matrix3x3<T> >
    operator*(U b, matrix3x3<T> a) {
        return a *= b, a;
    }

    // transpose
    template <typename T>
    constexpr matrix3x3<T> transpose(const matrix3x3<T>& a) {
        auto* r = a.rows;
        return {
            point3<T> { r[0].x, r[1].x, r[2].x },
            point3<T> { r[0].y, r[1].y, r[2].y },
            point3<T> { r[0].z, r[1].z, r[2].z },
        };
    }

    // A * b where b is assumed to be a 3x1 column
    // and the result is a 3x1 column
    template <typename T, typename U>
    constexpr auto operator*(matrix3x3<T> a, const point3<U>& b) {
        using R = decltype(dot(a.rows[0],b));
        return point3<R> {
            dot(a.rows[0],b), dot(a.rows[1],b), dot(a.rows[2],b)
        };
    }
}
