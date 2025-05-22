#pragma once

#include <opencv2/core/core.hpp>

namespace cv {
    template <typename T>
    inline cv::Point_<T> rotate_clockwise_90(const cv::Point_<T>& p) {
        return { -p.y, p.x };
    }

    template <typename T>
    cv::Point2f midpoint(const Point_<T>& a, const Point_<T>& b) {
        return { float(a.x+b.x)/2, float(a.y+b.y)/2 };
    }

    template <typename PT>
    inline cv::Point2f make2f(const PT& p) {
        return { float(p.x), float(p.y) };
    }
}
