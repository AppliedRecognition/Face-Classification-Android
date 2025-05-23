#pragma once

#include <cmath>

// note: stdlib log2() is chosen over this template if available
template <typename T>
static inline T log2(T x) { return 1.44269504 * std::log(x); }

namespace det {
    template <typename T>
    constexpr inline auto sqr(T x) -> decltype(x*x) {
        return x*x;
    }
    constexpr inline float raddeg(float degrees) {
        return degrees * float(M_PI/180);
    }
    constexpr inline double raddeg(double degrees) {
        return degrees * (M_PI/180);
    }
    inline float sindeg(float degrees) {
        return std::sin(raddeg(degrees));
    }
    inline double sindeg(double degrees) {
        return std::sin(raddeg(degrees));
    }
    inline float cosdeg(float degrees) {
        return std::cos(raddeg(degrees));
    }
    inline double cosdeg(double degrees) {
        return std::cos(raddeg(degrees));
    }
    inline float tandeg(float degrees) {
        return std::tan(raddeg(degrees));
    }
    inline double tandeg(double degrees) {
        return std::tan(raddeg(degrees));
    }
    inline double atan2deg(double dy, double dx) {
        return std::atan2(dy, dx) * (180/M_PI);
    }
    inline double log2d(double x) { return log2(x); }
}
