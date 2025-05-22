
#include "frontalize.hpp"
#include "opencv_mesh.hpp"
#include "dlib.hpp"
#include "diagnostics.hpp"

#include <raw_image/opencv.hpp>
#include <raw_image/point_rounding.hpp>
#include <core/context.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>

#include <applog/core.hpp>

#include <cmath>

#include "frontalize_model.ipp"


using namespace render;


static const std::vector<cv::Point3f>& face_center() {
    static const auto r = [] {
        std::vector<cv::Point> poly;
        for (unsigned i = 0; i <= 16; ++i)
            poly.emplace_back(
                depth_center_x + cvRound(landmark_3d[i].x),
                depth_center_y + cvRound(landmark_3d[i].y));
        poly.emplace_back(
            depth_center_x + cvRound(landmark_3d[26].x),
            depth_center_y + cvRound(landmark_3d[26].y));
        poly.emplace_back(
            depth_center_x + cvRound(landmark_3d[17].x),
            depth_center_y + cvRound(landmark_3d[17].y));
        const auto first_y = poly.back().y;
        const auto last_y = poly[8].y + 1;
        assert(last_y > first_y);
        std::vector<cv::Point3f> r;
        r.reserve(unsigned(depth_img.cols*(last_y-first_y)));
        for (int y = first_y; y < last_y; ++y) {
            const auto fy = float(y - depth_center_y);
            auto dx = depth_img.ptr<unsigned char>(y);
            for (int x = 0; x < depth_img.cols; ++x, ++dx) {
                if (95 < *dx && *dx < 200 &&
                    pointPolygonTest(poly, {float(x),float(y)}, false) > 0) {
                    const auto fx = float(x - depth_center_x);
                    r.emplace_back(fx, fy, 255 - *dx);
                }
            }
        }
        //FILE_LOG(logINFO) << "central face region: " << r.size() << " points";
        return r;
    }();
    assert(!r.empty());
    return r;
}

static const detected_coordinates& frontal_landmarks() {
    static const auto r = [] {
        detected_coordinates dc(det::dt::dlib68);
        dc.landmarks.reserve(68);
        for (unsigned i = 0; i < 68; ++i) {
            const auto x = depth_center_x + landmark_3d[i].x;
            const auto y = depth_center_y + landmark_3d[i].y;
            dc.landmarks.push_back({x,y});
        }
        dc.eye_left  = 0.5f * (dc.landmarks[36] + dc.landmarks[39]);
        dc.eye_right = 0.5f * (dc.landmarks[42] + dc.landmarks[45]);
        dc.confidence = 1;
        return dc;
    }();
    return r;
}

// find point on line segment a-b closest to p
static cv::Point2f
project_to_line(const cv::Point2f& p, const cv::Point2f& a, cv::Point2f b) {
    b -= a;
    const auto d = b.dot(b);
    if (d <= 0) return a;
    const auto x = b.dot(p-a) / d;
    if (x <= 0) return a;
    if (x >= 1) return b;
    return a + x*b;
}

static const std::vector<cv::Point3f>& triangle_vertices() {
    static const auto r = [] {
        assert(depth_img.rows > 0 &&
               depth_img.cols > 0 && (depth_img.cols&1) == 0);
        std::vector<cv::Point3f> pts;
        pts.reserve(std::size_t(depth_img.rows*depth_img.cols)/2);
        for (int y = 1; y < depth_img.rows; ++y) {
            const auto fy = float(y - depth_center_y) - 0.5f;
            auto pz0 = depth_img.ptr<unsigned char>(y-1) + (y&1);
            auto pz1 = depth_img.ptr<unsigned char>(y)   + (y&1);
            for (int x = y&1; x < depth_img.cols; x+=2, pz0+=2, pz1+=2)
                pts.emplace_back(
                    x - depth_center_x, fy, float(510-*pz0-*pz1)/2.0f);
        }
        assert(2*pts.size() == std::size_t(depth_img.rows-1) * std::size_t(depth_img.cols));
        return pts;
    }();
    return r;
}

// indices to trangle vertices
static std::array<unsigned,3> triangle_indices(unsigned i) {
    static const auto uc = unsigned(depth_img.cols);
    const auto y = i / uc;
    assert(y > 0);
    const auto x = i % uc;
    const auto row1 = y * (uc/2);
    const auto row0 = row1 - uc/2;
    std::array<unsigned,3> v;
    if ((x^y)&1) {
        v[0] = row0 + (x-1)/2;
        v[1] = row0 + (x+1)/2;
        v[2] = row1 + x/2;
    }
    else {
        v[0] = row0 + x/2;
        v[1] = row1 + (x+1)/2;
        v[2] = row1 + (x-1)/2;
    }
    return v;
}

template <typename T>
static void vis_dilate(cv::Mat_<T>& img, cv::Point& p) {
    if (img(p.y,p.x) <= 0) {
        const auto c =
            (img(p.y-1,p.x-1) > 0) +
            (img(p.y-1,p.x  ) > 0) +
            (img(p.y-1,p.x+1) > 0) +
            (img(p.y,  p.x-1) > 0) +
            (img(p.y,  p.x+1) > 0) +
            (img(p.y+1,p.x-1) > 0) +
            (img(p.y+1,p.x  ) > 0) +
            (img(p.y+1,p.x+1) > 0);
        if (c > 4) {
            img(p.y,p.x) = 4;
            if (!--p.x) p.x = 1;
            if (!--p.y) p.y = 1;
        }
    }
}

template <typename T>
static void vis_erode(cv::Mat_<T>& img, cv::Point& p) {
    if (img(p.y,p.x) > 0) {
        const auto c =
            (img(p.y-1,p.x-1) > 0) +
            (img(p.y-1,p.x  ) > 0) +
            (img(p.y-1,p.x+1) > 0) +
            (img(p.y,  p.x-1) > 0) +
            (img(p.y,  p.x+1) > 0) +
            (img(p.y+1,p.x-1) > 0) +
            (img(p.y+1,p.x  ) > 0) +
            (img(p.y+1,p.x+1) > 0);
        if (c < 4) {
            img(p.y,p.x) = 0;
            if (!--p.x) p.x = 1;
            if (!--p.y) p.y = 1;
        }
    }
}

template <typename T>
static void vis_smooth(cv::Mat_<T>& img, cv::Point& p) {
    static const auto set_min = [](T& a, const T& b) { if (a > b) a = b; };
    if (auto c = img(p.y,p.x)) {
        set_min(c, img(p.y-1,p.x-1));
        set_min(c, img(p.y-1,p.x  ));
        set_min(c, img(p.y-1,p.x+1));
        set_min(c, img(p.y,  p.x-1));
        set_min(c, img(p.y,  p.x+1));
        set_min(c, img(p.y+1,p.x-1));
        set_min(c, img(p.y+1,p.x  ));
        set_min(c, img(p.y+1,p.x+1));
        if (img(p.y,p.x) > ++c) {
            img(p.y,p.x) = c;
            if (!--p.x) p.x = 1;
            if (!--p.y) p.y = 1;
        }
    }
}

template <typename T>
static void vis_3x3(cv::Mat_<T>& img, void(*fn)(cv::Mat_<T>&,cv::Point&)) {
    for (cv::Point p(1,1); p.y < img.rows-1; ++p.y)
        for (p.x = 1; p.x < img.cols-1; ++p.x)
            fn(img, p);
}


namespace {
    struct projection {
        using mat_type = cv::Mat_<float>;
        const mat_type camera;
        const mat_type translation;
        const mat_type rotation;
        mat_type proj_mat;
        
        explicit projection(const face_alignment& a)
            : camera((mat_type(3,3) <<
                      a.focal_length, 0, a.image_center.x,
                      0, a.focal_length, a.image_center.y,
                      0, 0, 1)),
              translation((mat_type(3,1) << a.tx, a.ty, a.tz)),
              rotation((mat_type(3,1) <<
                        a.pitch * (M_PI/180),
                        a.yaw * (M_PI/180),
                        a.roll * (M_PI/180))),
              proj_mat([&]{
                      cv::Mat rm;
                      Rodrigues(rotation, rm);
                      mat_type rt(3,4);
                      rm.copyTo(rt.colRange(0,3));
                      translation.copyTo(rt.col(3));
                      return camera * rt;
                  }()) {}

        std::vector<cv::Point2f>
        project2(const cv::Point3f* first, const cv::Point3f* last) const {
            const auto n = stdx::round_to<int>(last - first);
            const cv::Mat pt3(n, 3, CV_32F, const_cast<cv::Point3f*>(first));
            mat_type pt4(n, 4);
            pt3.copyTo(pt4.colRange(0,3));
            pt4.col(3) = 1.0f;
            
            const auto X = mat_type(proj_mat * pt4.t());
            assert(X.rows == 3 && X.cols >= 0);
            
            std::vector<cv::Point2f> r;
            r.reserve(unsigned(X.cols));
            for (int i = 0; i < X.cols; ++i) {
                const auto z = X(2,i);
                r.emplace_back(X(0,i) / z, X(1,i) / z);
            }
            return r;
        }
        std::vector<cv::Point2f>
        project2(const std::vector<cv::Point3f>& pts) const {
            return project2(pts.data(), pts.data() + pts.size());
        }
        
        std::vector<cv::Point3f>
        project3(const cv::Point3f* first, const cv::Point3f* last) const {
            const auto n = stdx::round_to<int>(last - first);
            const cv::Mat pt3(n, 3, CV_32F, const_cast<cv::Point3f*>(first));
            mat_type pt4(n, 4);
            pt3.copyTo(pt4.colRange(0,3));
            pt4.col(3) = 1.0f;
            
            const auto X = mat_type(proj_mat * pt4.t());
            assert(X.rows == 3 && X.cols >= 0);
            
            std::vector<cv::Point3f> r;
            r.reserve(unsigned(X.cols));
            for (int i = 0; i < X.cols; ++i) {
                const auto z = X(2,i);
                r.emplace_back(X(0,i) / z, X(1,i) / z, z);
            }
            return r;
        }
        std::vector<cv::Point3f>
        project3(const std::vector<cv::Point3f>& pts) const {
            return project3(pts.data(), pts.data() + pts.size());
        }

        template <typename PT>
        std::vector<cv::Point2f>
        project_landmarks(const std::vector<PT>& pts2d,
                          std::vector<cv::Point2f>* hullp = nullptr) const {
            
            auto r = project2(landmark_3d, landmark_3d + 68);

            std::vector<cv::Point2f> hull;
            convexHull(project2(face_center()), hull);
            assert(hull.size() > 2);
            
            const auto norm_sqr = [](const cv::Point2f& a) {
                return a.x*a.x + a.y*a.y;
            };
            assert(pts2d.size() == 68);

            // move any jaw-line landmark inside hull to edge
            for (unsigned i = 0; i <= 16; ++i)
                if (pointPolygonTest(hull, r[i], false) > 0) {
                    const auto target = raw_image::round_to<cv::Point2f>(pts2d[i]);
                    auto p0 = hull.back();
                    auto best_p = p0;
                    auto best_e = norm_sqr(target - best_p);
                    for (auto& p1 : hull) {
                        const auto p = project_to_line(target, p0, p1);
                        const auto e = norm_sqr(target - p);
                        if (best_e > e) {
                            best_e = e;
                            best_p = p;
                        }
                        p0 = p1;
                    }
                    r[i] = best_p;
                }
            
            if (hullp) *hullp = move(hull);
            return r;
        }

        cv::Mat render_frontal(const cv::Mat& img, const cv::Point& ofs) const {
            cv::Mat dest(depth_img.size(), img.type(), cvScalarAll(0));
            mat_type pt4(4,1);
            pt4(3) = 1;
            cv::Rect sp(0,0,1,1);
            for (int y = 0; y < depth_img.rows; ++y) {
                pt4(1) = float(y - depth_center_y);
                auto dx = depth_img.ptr<unsigned char>(y);
                auto dest_row = dest.row(y);
                for (int x = 0; x < depth_img.cols; ++x, ++dx) {
                    if (*dx > 0) {
                        pt4(0) = float(x - depth_center_x);
                        pt4(2) = float(255 - *dx);
                        const auto X = mat_type(proj_mat * pt4);
                        const auto z = X(2);
                        sp.x = cvRound(X(0)/z) - ofs.x;
                        sp.y = cvRound(X(1)/z) - ofs.y;
                        if (sp.x >= 0 && sp.x < img.cols &&
                            sp.y >= 0 && sp.y < img.rows)
                            img(sp).copyTo(dest_row.col(x));
                    }
                }
            }
            return dest;
        }

        cv::Mat render_visibility() const {
            // project points
            auto pts = project3(triangle_vertices());
            const auto triangle = &triangle_indices;

            // sort triangles by -z (furthest from camera first)
            std::vector<std::pair<float, unsigned> > tri_idx;
            tri_idx.reserve(std::size_t(depth_img.rows*depth_img.cols));
            for (int y = 1; y < depth_img.rows; ++y) {
                auto pz = depth_img.ptr<unsigned char>(y);
                auto i = unsigned(y * depth_img.cols);
                for (int x = 0; x < depth_img.cols-1; ++x, ++pz, ++i)
                    if (*pz) {
                        const auto t = triangle_indices(i);
                        const auto z = pts[t[0]].z + pts[t[1]].z + pts[t[2]].z;
                        assert(z > 0);
                        tri_idx.emplace_back(-z, i);
                    }
            }
            std::sort(tri_idx.begin(), tri_idx.end());
            //FILE_LOG(logINFO) << "triangles: " << tri_idx.size();

            // normalize 2d range to box [0,0] x [1,1]
            auto minp = pts[triangle(tri_idx.front().second)[0]], maxp = minp;
            for (const auto& ti : tri_idx) {
                for (auto i : triangle_indices(ti.second)) {
                    const auto& p = pts[i];
                    if (minp.x > p.x) minp.x = p.x;
                    if (minp.y > p.y) minp.y = p.y;
                    if (maxp.x < p.x) maxp.x = p.x;
                    if (maxp.y < p.y) maxp.y = p.y;
                }
            }
            for (auto& p : pts) {
                p.x = (p.x-minp.x) / (maxp.x-minp.x);
                p.y = (p.y-minp.y) / (maxp.y-minp.y);
            }

            // index map
            static constexpr auto ph = 512;
            static constexpr auto pw = 512;
            cv::Mat index_map(ph,pw,CV_8UC3,cvScalarAll(0));

            // render triangles (furthest from camera first)
            for (const auto& ti : tri_idx) {
                const auto t = triangle_indices(ti.second);
                cv::Point pt[3];
                for (unsigned i = 0; i < 3; ++i) {
                    pt[i].x = cvRound((pw-1)*pts[t[i]].x);
                    pt[i].y = cvRound((ph-1)*pts[t[i]].y);
                    assert(pt[i].x >= 0 && pt[i].x < pw);
                    assert(pt[i].y >= 0 && pt[i].y < ph);
                }
                const cv::Point* pp = pt;
                const int n = 3;
                // encode triangle index as color
                const auto c = cvScalar(
                    ti.second&255, (ti.second>>8)&255, (ti.second>>16)&255);
                fillPoly(index_map, &pp, &n, 1, c);
            }

            // tally visible pixels
            std::vector<unsigned> tally(
                std::size_t(depth_img.rows*depth_img.cols), 0);
            for (int y = 0; y < index_map.rows; ++y) {
                auto ptr = index_map.ptr<unsigned char>(y);
                for (int x = 0; x < index_map.cols; ++x, ptr+=3) {
                    if (auto j = ptr[0] + unsigned(ptr[1]<<8) + unsigned(ptr[2]<<16)) {
                        assert(j < tally.size());
                        ++tally[j];
                    }
                }
            }

            cv::Mat_<unsigned char> vis(depth_img.size());
            vis = 0;

            for (unsigned i = 0; i < tally.size(); ++i)
                if (tally[i]) vis(int(i)) = 4;

            vis_3x3(vis,&vis_dilate);
            vis_3x3(vis,&vis_erode);
            vis_3x3(vis,&vis_smooth);
            
            vis *= 64;
            return vis;
        }

    };
}

std::pair<raw_image::plane_ptr,coordinate_type>
render::render_model(stdx::arg<core::context_data> context,
                     const face_alignment& alignment) {
    if (!context)
        throw std::invalid_argument("invalid context object");

    const auto fit = projection(alignment);

    static const auto model_pts = [] {
        std::vector<cv::Point3f> pts;
        pts.reserve(std::size_t(depth_img.rows*depth_img.cols));
        for (int y = 0; y < depth_img.rows; ++y) {
            const auto fy = y - depth_center_y;
            auto dx = depth_img.ptr<unsigned char>(y);
            for (int x = 0; x < depth_img.cols; ++x, ++dx)
                if (*dx > 0)
                    pts.emplace_back(x - depth_center_x, fy, 255 - *dx);
        }
        return pts;
    }();
    const auto proj_pts = fit.project3(model_pts);

    auto minp = proj_pts.front(), maxp = minp;
    for (const auto& p : proj_pts) {
        if (minp.x > p.x) minp.x = p.x;
        if (maxp.x < p.x) maxp.x = p.x;
        if (minp.y > p.y) minp.y = p.y;
        if (maxp.y < p.y) maxp.y = p.y;
        if (minp.z > p.z) minp.z = p.z;
        if (maxp.z < p.z) maxp.z = p.z;
    }
    assert(minp.z > 0);

    const coordinate_type ofs = { std::floor(minp.x), std::floor(minp.y) };
    const auto w = unsigned(cvRound(1 + std::ceil(maxp.x) - ofs.x));
    const auto h = unsigned(cvRound(1 + std::ceil(maxp.y) - ofs.y));
    auto r = raw_image::create(w, h, raw_image::pixel::gray8);
    cv::Mat_<unsigned char> rm = to_Mat(r);
    rm = 0;
    
    for (const auto& p : proj_pts) {
        const auto x = cvRound(p.x - ofs.x);
        const auto y = cvRound(p.y - ofs.y);
        assert (x >= 0 && x < rm.cols && y >= 0 && y < rm.rows);
        const auto z =
            cv::saturate_cast<unsigned char>(256*(maxp.z-p.z)/(maxp.z-minp.z));
        if (rm(y,x) < z)
            rm(y,x) = z;
    }
    
    return { move(r), ofs };
}

static void inplace_push_border(std::vector<cv::Point>& pts) {
    assert(pts.size() == 68);

    const auto bp = raw_image::round_to<cv::Point2d>(pts.front());
    const auto v = pts[16] - pts[0];
    const auto n = norm(v);
    const auto nd8 = n / 8;
    const auto r = cv::Point2d(v.x / n, v.y / n);

    std::vector<cv::Point> top;
    top.reserve(16);
    
    for (unsigned i = 0; i <= 16; ++i) {
        cv::Point tang;
        if (i == 0)
            tang = pts[1] - pts[0];
        else if (i == 16)
            tang = pts[16] - pts[15];
        else
            tang = pts[i+1] - pts[i-1];
        const auto s = nd8 / norm(tang);
        const auto p = cv::Point2d(pts[i].x - s*tang.y, pts[i].y + s*tang.x);
        pts.push_back(raw_image::round_from(p));

        if (i > 0 && i < 16) {
            const auto q = bp + r.dot(p - bp) * r;
            const auto u = 2*q - p;
            top.push_back(raw_image::round_from(u));
        }
    }

    pts.insert(pts.end(), top.rbegin(), top.rend());
}

static const auto eye_left  = 0.5f * (landmark_3d[36] + landmark_3d[39]);
static const auto eye_right = 0.5f * (landmark_3d[42] + landmark_3d[45]);

std::pair<raw_image::plane_ptr,raw_image::plane_ptr>
render::render_frontal(stdx::arg<core::context_data> context,
                       const face_coordinates& detected_face,
                       stdx::arg<const raw_image::plane> image,
                       const face_alignment& alignment,
                       const render_settings& rsettings,
                       const output_settings& osettings,
                       diagnostics* diag) {

    if (!context)
        throw std::invalid_argument("invalid context object");
    if (!image || !image->data)
        throw std::invalid_argument("invalid image object");
    if ((image->rotate&7) || image->scale)
        throw std::runtime_error("rotate/scale not implemented");

    const auto fit = projection(alignment);

    const det::detected_coordinates* shape = nullptr;
    for (auto& s : detected_face)
        if (s.type == det::dt::dlib68) {
            shape = &s;
            break;
        }
    if (!shape || shape->landmarks.size() != 68)
        throw std::invalid_argument("dlib landmarks required");

    const auto proj2d = fit.project_landmarks(shape->landmarks);
    assert(proj2d.size() == shape->landmarks.size());

    std::vector<cv::Point> src_pts, dest_pts;
    src_pts.reserve(68 + 32);
    dest_pts.reserve(68 + 32);
    for (unsigned i = 0; i < proj2d.size(); ++i) {
        src_pts.push_back(raw_image::round_from(shape->landmarks[i]));
        dest_pts.push_back(raw_image::round_from(proj2d[i]));
    }
    inplace_push_border(src_pts);
    inplace_push_border(dest_pts);

    const auto dest_dim = boundingRect(dest_pts);
    for (auto& p : dest_pts)
        p -= dest_dim.tl();
    const auto mesh = cvx::compute_mesh(dest_pts);

    const auto src_img = to_Mat(*image);
    auto dest_img = cv::Mat(dest_dim.size(), src_img.type(), cvScalarAll(0));
    cvx::warp_mesh(dest_img, dest_pts, src_img, src_pts, mesh);
    auto dest_raw = to_raw_image(dest_img);
    dest_raw.layout = image->layout;
    if (bytes_per_pixel(osettings.color_space) == 1) {
        auto p = convert(dest_raw, raw_image::pixel::gray8);
        assert(!p);
    }
    else if (bytes_per_pixel(dest_raw.layout) != 1) {
        auto p = convert(dest_raw, raw_image::pixel::yuv);
        assert(!p);
    }
    const auto frontal = fit.render_frontal(to_Mat(dest_raw), dest_dim.tl());
    auto f_raw = to_raw_image(frontal);
    if (frontal.type() == CV_8UC3)
        f_raw.layout = raw_image::pixel::yuv;

    const auto vis = fit.render_visibility();
    
    internal::in_place_correct_lighting_dlib(
        f_raw, to_raw_image(vis), frontal_landmarks(), rsettings, diag);
    
    assert(fabsf(eye_left.x + eye_right.x) < 1e-5);
    assert(fabsf(eye_left.y - eye_right.y) < 1e-5);
    assert(fabsf(eye_left.z - eye_right.z) < 1e-5);

    if (diag) {
        const auto scale =
            (float(osettings.width) * osettings.eye_width) / eye_right.x / 2;
        const auto ofsx = float(osettings.width) / 2;
        const auto ofsy = float(osettings.height) * osettings.eye_vertical;
        diag->final_landmarks.clear();
        diag->final_landmarks.reserve(68);
        for (unsigned i = 0; i < 68; ++i) {
            const auto x = cvRound(ofsx + scale * landmark_3d[i].x);
            const auto y = cvRound(ofsy + scale * (landmark_3d[i].y - eye_right.y));
            diag->final_landmarks.emplace_back(x,y);
        }
    }
    
    const auto scale =
        2 * eye_right.x / (float(osettings.width) * osettings.eye_width);
    const auto ofsx = depth_center_x - eye_right.x / osettings.eye_width;
    const auto ofsy = depth_center_y + eye_right.y
        - float(osettings.height) * osettings.eye_vertical * scale;
    const cv::Mat_<float> M =
        (cv::Mat_<float>(2,3) << scale, 0, ofsx, 0, scale, ofsy);

    const auto finish = [&](const cv::Mat& img) {
        auto r = raw_image::create(
            osettings.width, osettings.height,
            img.type() == CV_8UC1 ?
            raw_image::pixel::gray8 : raw_image::pixel::yuv);
        auto dest = to_Mat(r);
        warpAffine(img, dest, M, dest.size(),
                   cv::WARP_INVERSE_MAP + cv::INTER_LINEAR);
        return r;
    };
    
    auto r = finish(to_Mat(f_raw));
    if (auto u = convert(*r, osettings.color_space))
        r = move(u);
    
    return { move(r), finish(vis) };
}

void render::mask_visibility(raw_image::plane& image,
                             stdx::arg<const raw_image::plane> visibility,
                             unsigned threshold,
                             std::array<unsigned char,4> color) {
    if (!visibility || bytes_per_pixel(visibility->layout) != 1)
        throw std::invalid_argument("invalid visibility image");
    if (image.width != visibility->width ||
        image.height != visibility->height ||
        image.rotate != visibility->rotate ||
        image.scale != visibility->scale)
        throw std::invalid_argument("visibility image has wrong size");
    const auto bpp = bytes_per_pixel(image);
    for (unsigned y = 0; y < image.height; ++y) {
        auto m = visibility->data + y * visibility->bytes_per_line;
        auto p = image.data + y * image.bytes_per_line;
        for (unsigned x = 0; x < image.width; ++x, ++m, p += bpp)
            if (*m < threshold)
                memcpy(p, color.data(), bpp);
    }
}
