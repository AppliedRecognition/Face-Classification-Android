
#include "opencv_mesh.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <applog/core.hpp>

#include <map>


using namespace cvx;


template <typename T>
constexpr inline auto sqr(T x) -> decltype(x*x) {
    return x*x;
}

namespace cv {
    static bool operator<(const Point& a, const Point& b) {
        return a.x <= b.x && (a.x < b.x || a.y < b.y);
    }
}

// returns 0 in degenerate cases
template <typename PT>
static int triangle_sense(const PT& p0, const PT& p1, const PT& p2) {
    const auto r = (p1.x-p0.x)*(p2.y-p0.y) - (p1.y-p0.y)*(p2.x-p0.x);
    if (r > 0) return 1;
    if (r < 0) return -1;
    return 0;
}

// copy pixels within triangle from src image to tmp buffer
// tmp_buf and tmp_mask must be pre-allocated and of sufficient size
// returns roi within dest image
// ie. copy pixels from tmp_buf(rect(0,0,roi.w,roi.h)) to dest_img(roi)
//     subject to tmp_mask(rect(0,0,roi.w,roi.h))
static cv::Rect warp_triangle(
    const cv::Mat& src_img, const cv::Scalar& src_border,
    const cv::Point src_pts[3],
    const cv::Mat& tmp_buf, const cv::Mat& tmp_mask,
    const cv::Point dest_pts[3]) {

    assert(tmp_buf.type() == src_img.type());

    const auto dest_rect = [&] {
        auto r = cv::Rect(
            std::min({dest_pts[0].x,dest_pts[1].x,dest_pts[2].x}),
            std::min({dest_pts[0].y,dest_pts[1].y,dest_pts[2].y}),
            std::max({dest_pts[0].x,dest_pts[1].x,dest_pts[2].x}) + 1,
            std::max({dest_pts[0].y,dest_pts[1].y,dest_pts[2].y}) + 1);
        r.width  -= r.x;
        r.height -= r.y;
        return r;
    }();
    if (tmp_buf.cols  < dest_rect.width || tmp_buf.rows  < dest_rect.height ||
        tmp_mask.cols < dest_rect.width || tmp_mask.rows < dest_rect.height)
        throw std::invalid_argument("destination has insufficient size");
    const auto tmp_rect = cv::Rect({0,0},dest_rect.size());
    auto mask = tmp_mask(tmp_rect);
    mask.setTo(0);
    
    const auto ds = triangle_sense(dest_pts[0], dest_pts[1], dest_pts[2]);
    if (ds == 0) {
        FILE_LOG(logWARNING) << "not copying to degenerate triangle";
        return dest_rect;
    }

    const cv::Point tmp_pts[3] = {
        dest_pts[0] - dest_rect.tl(),
        dest_pts[1] - dest_rect.tl(),
        dest_pts[2] - dest_rect.tl()
    };
    const cv::Point2f tpf[] = { tmp_pts[0], tmp_pts[1], tmp_pts[2] };
    
    {  // create mask
        const auto* v = tmp_pts;
        static constexpr auto n = 3;
        fillPoly(mask, &v, &n, 1, cvScalarAll(1));
    }
    auto buf = tmp_buf(tmp_rect);

    cv::Point2f spf[] = { src_pts[0], src_pts[1], src_pts[2] };
    if (triangle_sense(src_pts[0], src_pts[1], src_pts[2]) == 0) {
        // source triangle is line or single pixel
        const auto v01 = src_pts[0] - src_pts[1];
        const auto d01 = sqr(v01.x) + sqr(v01.y);
        const auto v02 = src_pts[0] - src_pts[2];
        const auto d02 = sqr(v02.x) + sqr(v02.y);
        const auto v12 = src_pts[1] - src_pts[2];
        const auto d12 = sqr(v12.x) + sqr(v12.y);
        if (d01 >= std::max(d02,d12)) {
            if (d01 == 0) {  // single pixel
                resize(src_img(cv::Rect(src_pts[0],cv::Size(1,1))),
                       buf, buf.size(), cv::INTER_NEAREST);
                return dest_rect;
            }
            else {
                // move spf[2] perp to line to make triangle
                const auto d = std::sqrt(d01);
                const auto v = cv::Point2f(float(v01.y/d), float(-v01.x/d));
                if (ds * triangle_sense(spf[0], spf[1], spf[2]+v) >= 0)
                    spf[2] += v;
                else
                    spf[2] -= v;
            }
        }
        else if (d02 >= std::max(d01,d12)) {
            // move spf[1] perp to line to make triangle
            const auto d = std::sqrt(d02);
            const auto v = cv::Point2f(float(v02.y/d), float(-v02.x/d));
            if (ds * triangle_sense(spf[0], spf[1]+v, spf[2]) >= 0)
                spf[1] += v;
            else
                spf[1] -= v;
        }
        else { // d12 >= std::max(d01,d02)
            // move spf[0] perp to line to make triangle
            const auto d = std::sqrt(d12);
            const auto v = cv::Point2f(float(v12.y/d), float(-v12.x/d));
            if (ds * triangle_sense(spf[0]+v, spf[1], spf[2]) >= 0)
                spf[0] += v;
            else
                spf[0] -= v;
        }
    }

    assert(triangle_sense(spf[0], spf[1], spf[2]) != 0);

    const auto t = getAffineTransform(spf, tpf);
    warpAffine(src_img, buf, t, buf.size(),
               cv::INTER_CUBIC, cv::BORDER_CONSTANT, src_border);
    
    return dest_rect;
}

template <int n>
static void blendTo(const cv::Mat& src, const cv::Mat_<uint8_t>& mask,
                    cv::Mat dest, cv::Mat_<uint8_t> weight) {
    assert(src.channels() == n);
    assert(src.depth() == CV_8U);
    assert(dest.type() == src.type());

    assert(mask.size() == src.size());
    assert(dest.size() == src.size());
    assert(weight.size() == src.size());

    for (int y = 0; y < src.rows; ++y) {
        auto sp = src.ptr<const uint8_t>(y);
        auto mp = mask.ptr<const uint8_t>(y);
        auto dp = dest.ptr<uint8_t>(y);
        auto wp = weight.ptr<uint8_t>(y);
        for (auto i = src.cols; i > 0; --i, sp += n, ++mp, dp += n, ++wp) {
            if (!*mp) continue;
            if (*wp <= 0) {
                std::copy_n(sp, n, dp);
                *wp = 1;
            }
            else if (*wp < 255) {
                const auto w = *wp;
                const auto w1 = *wp = static_cast<uint8_t>(w + 1);
                const auto h = w1 / 2;
                for (int i = 0; i < n; ++i)
                    dp[i] = static_cast<uint8_t>((sp[i] + w*dp[i] + h) / w1);
            }
        }
    }
}

void cvx::warp_mesh(cv::Mat dest_img, const cv::Point* dest_pts,
                    const cv::Mat& src_img, const cv::Point* src_pts,
                    stdx::forward_iterator<const triangle_type&> first,
                    stdx::forward_iterator<const triangle_type&> last,
                    std::size_t N) {
    if (dest_img.type() != src_img.type())
        throw std::invalid_argument("src and dest images must have same type");
    assert(dest_pts);
    assert(src_pts);

    const auto blend = [](int nc) {
        switch(nc) {
        case 1: return &blendTo<1>;
        case 3: return &blendTo<3>;
        default: throw std::invalid_argument("image must be 1 or 3 channels");
        }
    }(src_img.channels());

    auto dest_roi = cv::Rect(dest_img.cols,dest_img.rows,0,0);
    auto tmp_size = cv::Size(0,0);
    for (auto it = first; it != last; ++it) {
        auto& t = *it;
        if (t.size() != 3 || t[0] >= t[1] || t[1] >= t[2] || t[2] >= N)
            throw std::invalid_argument("malformed triangle");
        const auto lx =
            std::min({dest_pts[t[0]].x, dest_pts[t[1]].x, dest_pts[t[2]].x});
        const auto ty =
            std::min({dest_pts[t[0]].y, dest_pts[t[1]].y, dest_pts[t[2]].y});
        const auto rx =
            std::max({dest_pts[t[0]].x, dest_pts[t[1]].x, dest_pts[t[2]].x});
        const auto by =
            std::max({dest_pts[t[0]].y, dest_pts[t[1]].y, dest_pts[t[2]].y});
        if (tmp_size.width <= rx-lx)
            tmp_size.width = rx-lx+1;
        if (tmp_size.height <= by-ty)
            tmp_size.height = by-ty+1;
        if (dest_roi.x > lx)
            dest_roi.x = lx;
        if (dest_roi.y > ty)
            dest_roi.y = ty;
        if (dest_roi.width <= rx)
            dest_roi.width = rx+1;
        if (dest_roi.height <= by)
            dest_roi.height = by+1;
    }
    dest_roi.width  -= dest_roi.x;
    dest_roi.height -= dest_roi.y;
    if (dest_roi.x < 0) {
        dest_roi.width += dest_roi.x;
        if (dest_roi.width <= 0) return;
        dest_roi.x = 0;
    }
    if (dest_roi.y < 0) {
        dest_roi.height += dest_roi.y;
        if (dest_roi.height <= 0) return;
        dest_roi.y = 0;
    }
    if (dest_roi.width + dest_roi.x > dest_img.cols) {
        dest_roi.width = dest_img.cols - dest_roi.x;
        if (dest_roi.width <= 0) return;
    }
    if (dest_roi.height + dest_roi.y > dest_img.rows) {
        dest_roi.height = dest_img.rows - dest_roi.y;
        if (dest_roi.height <= 0) return;
    }
    
    const auto tmp_img = cv::Mat(tmp_size, dest_img.type());
    const auto tmp_mask = cv::Mat_<unsigned char>(tmp_size);
    const auto weight =
        cv::Mat_<unsigned char>(cv::Mat::zeros(dest_roi.size(), CV_8UC1));
    
    for ( ; first != last; ++first) {
        auto& t = *first;
        const cv::Point dp[3] = {
            dest_pts[t[0]], dest_pts[t[1]], dest_pts[t[2]]
        };
        const cv::Point sp[3] = {
            src_pts[t[0]], src_pts[t[1]], src_pts[t[2]]
        };
        auto dest_rect = warp_triangle(
            src_img, cvScalarAll(128), sp,
            tmp_img, tmp_mask, dp);

        auto tx = 0, ty = 0;
        
        if (dest_rect.x < 0) {
            tx = -dest_rect.x;
            dest_rect.width += dest_rect.x;
            if (dest_rect.width <= 0) continue;
            dest_rect.x = 0;
        }
        if (dest_rect.y < 0) {
            ty = -dest_rect.y;
            dest_rect.height += dest_rect.y;
            if (dest_rect.height <= 0) continue;
            dest_rect.y = 0;
        }
        if (dest_rect.width + dest_rect.x > dest_img.cols) {
            dest_rect.width = dest_img.cols - dest_rect.x;
            if (dest_rect.width <= 0) continue;
        }
        if (dest_rect.height + dest_rect.y > dest_img.rows) {
            dest_rect.height = dest_img.rows - dest_rect.y;
            if (dest_rect.height <= 0) continue;
        }

        const auto tr = cv::Rect({tx,ty}, dest_rect.size());
        const auto wr = dest_rect - dest_roi.tl();
        blend(tmp_img(tr), tmp_mask(tr), dest_img(dest_rect), weight(wr));
    }
}


static std::set<unsigned>
make_triangle(std::set<std::set<unsigned> >& edges,
              const std::vector<cv::Point>& pts,
              const cv::Point& vp) {
    std::set<unsigned> result;
    for (auto it = edges.begin(), end = edges.end(); it != end; ++it) {
        assert(it->size() == 2);
        for (auto jt = next(it); jt != end; ++jt) {
            std::vector<unsigned> com;
            set_intersection(it->begin(), it->end(),
                             jt->begin(), jt->end(),
                             back_inserter(com));
            if (com.size() == 1) {
                result = *it;
                result.insert(jt->begin(), jt->end());
                assert(result.size() == 3);
                auto e = result;
                e.erase(com.front());
                auto kt = e.begin();
                const auto p0 = pts[*kt];
                const auto p1 = pts[*++kt];
                const auto p2 = pts[com.front()];
                const auto s0 = triangle_sense(p0, p1, p2);
                const auto s1 = triangle_sense(p0, p1, vp);
                if (s0*s1 < 0) {
                    edges.erase(it);
                    edges.erase(jt);
                    edges.insert(e);
                    return result;
                }
                result.clear();
            }
        }
    }
    return result;
}

static triangle_type triangle_from_set(const std::set<unsigned>& s) {
    assert(s.size() == 3);
    const auto i0 = s.begin();
    const auto i1 = next(i0);
    const auto i2 = next(i1);
    return { *i0, *i1, *i2 };
}

mesh_type cvx::compute_mesh(stdx::forward_iterator<cv::Point> pts_first,
                            stdx::forward_iterator<cv::Point> pts_last) {
    std::vector<cv::Point> pts;
    const auto N = distance(pts_first, pts_last);
    assert(N >= 0);
    pts.reserve(std::size_t(N));
    std::map<cv::Point, unsigned> point_map;
    cv::Rect rect(0,0,0,0);
    for ( ; pts_first != pts_last; ++pts_first) {
        const auto p = *pts_first;
        assert(p.x >= 0 && p.y >= 0);
        if (rect.width <= p.x)
            rect.width = p.x + 1;
        if (rect.height <= p.y)
            rect.height = p.y + 1;
        point_map.insert({p,pts.size()});
        pts.push_back(p);
    }

    cv::Subdiv2D subdiv(rect);
    for (const auto& p : point_map)
        subdiv.insert(p.first);
    std::vector<cv::Vec6f> tlist;
    subdiv.getTriangleList(tlist);  // delaunay

    mesh_type mesh;
    std::map<cv::Point, std::set<std::set<unsigned> > > extra;
    for (const auto& t : tlist) {
        std::set<unsigned> iset;
        cv::Point vp;
        const auto dp0 = cv::Point(cvRound(t[0]), cvRound(t[1]));
        const auto it0 = point_map.find(dp0);
        if (it0 != point_map.end())
            iset.insert(it0->second);
        else vp = dp0;
        const auto dp1 = cv::Point(cvRound(t[2]), cvRound(t[3]));
        const auto it1 = point_map.find(dp1);
        if (it1 != point_map.end())
            iset.insert(it1->second);
        else vp = dp1;
        const auto dp2 = cv::Point(cvRound(t[4]), cvRound(t[5]));
        const auto it2 = point_map.find(dp2);
        if (it2 != point_map.end())
            iset.insert(it2->second);
        else vp = dp2;
        if (iset.size() == 3) {
            if (triangle_sense(dp0, dp1, dp2))
                mesh.insert(triangle_from_set(iset));
        }
        else if (iset.size() == 2)
            extra[vp].insert(iset);
    }
    for (auto& p : extra) {
        for (;;) {
            const auto t = make_triangle(p.second, pts, p.first);
            if (!t.empty())
                mesh.insert(triangle_from_set(t));
            else
                break;
        }
    }
    return mesh;
}
