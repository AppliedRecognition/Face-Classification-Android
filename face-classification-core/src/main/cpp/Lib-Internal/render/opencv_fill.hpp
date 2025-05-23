#pragma once

#include <opencv2/core/core.hpp>

namespace cvx {
    class smooth_fill {
    public:
        const std::vector<std::pair<int,unsigned> >& mask;
        const int nrows;
        const int ncols;

    private:
        static constexpr const unsigned denom = 256;

        struct pixel_plan {
            cv::Point3i px[3];
            unsigned border = 0;
        };
        std::vector<pixel_plan> plan;

        void plan_pixel(int x, int y);

        unsigned char
        calc_pixel(const cv::Mat& img, const pixel_plan& p) const;
        
    public:
        smooth_fill(const std::vector<std::pair<int,unsigned> >& mask,
                    int ncols);

        void operator()(cv::Mat& img) const;
    };
}
