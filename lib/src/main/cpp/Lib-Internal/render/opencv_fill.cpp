
#include "opencv_fill.hpp"
#include <applog/core.hpp>


using namespace cvx;


namespace {
    template <typename T>
    constexpr inline auto sqr(T x) -> decltype(x*x) {
        return x*x;
    }

    struct sqr_dist_to {
        const int x, y;
        sqr_dist_to(int x, int y) : x(x), y(y) {}
        int operator()(int x0, int y0) const {
            return sqr(x-x0) + sqr(y-y0);
        }
    };
}

void smooth_fill::plan_pixel(int x, int y) {
    const sqr_dist_to d(x,y);
    std::vector<int> edge_dist = {
        d(-1, y-1),
        d(-1, y),
        d(-1, y+1),

        d(ncols, y-1),
        d(ncols, y),
        d(ncols, y+1),

        d(x-1, -1),
        d(x,   -1),
        d(x+1, -1),

        d(x-1, nrows),
        d(x,   nrows),
        d(x+1, nrows)
    };
    std::sort(edge_dist.begin(), edge_dist.end());
    assert(edge_dist.front() > 0);

    std::vector<cv::Point3i> inner_dist;
    inner_dist.reserve(unsigned(ncols) * mask.size());
    int y0 = 0;
    for (auto& m : mask) {
        for (int x0 = m.first; x0 < m.first + int(m.second); ++x0)
            inner_dist.push_back( { x0, y0, d(x0,y0) } );
        ++y0;
    }
    std::sort(inner_dist.begin(), inner_dist.end(),
              [](const cv::Point3i& a, const cv::Point3i& b) {
                  return a.z < b.z;
              });
    assert(inner_dist.front().z > 0);


    pixel_plan p;
    double df[3];
    double total = 0;
    for (unsigned i = 0; i < 3; ++i) {
        p.px[i] = { inner_dist[i].x, inner_dist[i].y, 0 };
        df[i] = 1 / sqrt(inner_dist[i].z);
        total += df[i];
        total += 1 / sqrt(edge_dist[i]);
    }
    p.border = denom;
    for (unsigned i = 0; i < 3; ++i) {
        p.px[i].z = cvRound(denom * df[i] / total);
        assert(p.px[i].z >= 0);
        p.border -= unsigned(p.px[i].z);
    }
    assert(p.border <= denom);
    plan.push_back(p);
}

unsigned char
smooth_fill::calc_pixel(const cv::Mat& img, const pixel_plan& p) const {
    unsigned z = 128 * p.border;
    for (unsigned i = 0; i < 3; ++i) {
        const auto v = *(img.ptr(p.px[i].y) + p.px[i].x);
        z += v * unsigned(p.px[i].z);
    }
    return (z / denom) & 0xff;
}

smooth_fill::smooth_fill(const std::vector<std::pair<int,unsigned> >& mask,
                         int ncols)
    : mask(mask), nrows(int(mask.size())), ncols(ncols) {
    assert(nrows > 0 && ncols > 0);
    int y = 0;
    for (auto& m : mask) {
        assert(m.first >= 0 &&
               unsigned(m.first) + m.second <= unsigned(ncols));
        for (int x = 0; x < m.first; ++x)
            plan_pixel(x,y);
        for (int x = m.first + int(m.second); x < ncols; ++x)
            plan_pixel(x,y);
        ++y;
    }
}

void smooth_fill::operator()(cv::Mat& img) const {
    assert(img.type() == CV_8UC1);
    assert(img.rows >= int(mask.size()) && img.cols >= ncols);
    int y = 0;
    auto it = plan.begin();
    for (auto& m : mask) {
        unsigned char* p = img.ptr(y);
        for (int x = 0; x < m.first; ++x, ++p, ++it)
            *p = calc_pixel(img, *it);
        p += m.second;
        for (int x = m.first + int(m.second); x < ncols; ++x, ++p, ++it)
            *p = calc_pixel(img, *it);
        ++y;
    }
    assert(it == plan.end());
}
