
#include "dlib.hpp"
#include "dlib_landmarks.hpp"
#include "dlib_mesh.ipp"
#include "dlib_lm3.ipp"
#include "dlib_pose.ipp"
#include "dlib_multipie.ipp"
#include "diagnostics.hpp"

#include "opencv_mesh.hpp"
#include "opencv_operators.hpp"

#include <det/math.hpp>
#include <det/landmark_standardize.hpp>

#include <raw_image/opencv.hpp>
#include <raw_image/points.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>

#include <applog/core.hpp>


using namespace render;
using det::sqr;


static void
inplace_correct_pose_0(cv::Mat_<float>& row, unsigned nvecs) {
    if (nvecs <= 0 || pm_vecs.rows <= 0) return;
    if (nvecs > unsigned(pm_vecs.rows))
        nvecs = unsigned(pm_vecs.rows);
    const auto vecs = pm_vecs.rowRange(0,int(nvecs));

    // subtract mean
    row -= pm_mean;

    // project
    const auto proj = cv::Mat_<float>(row * vecs.t());

    // remove reconstruction
    row -= proj * vecs;

    // add mean back in
    row += pm_mean;
}

static void
inplace_correct_pose(std::vector<cv::Point2f>& pts,
                     unsigned method, unsigned nvecs) {

    static_assert(sizeof(cv::Point2f) == 2*sizeof(float), "!!");
    assert(0 < pm_mean.cols && unsigned(pm_mean.cols) <= 2*pts.size());
    auto row = cv::Mat_<float>(1, pm_mean.cols,
                               reinterpret_cast<float*>(pts.data()),
                               cv::Mat::AUTO_STEP);

    switch (method) {
    case 0: inplace_correct_pose_0(row, nvecs); break;

    case 1: pm_mean.copyTo(row); break;

    case 2: mpie_neutral.copyTo(row); break;
    case 3: mpie_smile.copyTo(row); break;
    case 4: mpie_squint.copyTo(row); break;
    case 5: mpie_disgust.copyTo(row); break;
    case 6: mpie_surprise.copyTo(row); break;
    case 7: mpie_scream.copyTo(row); break;
        
    default:
        FILE_LOG(logWARNING) << "unknown pose_variant (no pose correction)";
    }
}

template <typename T>
static void inplace_zero_outside_mask(cv::Mat_<T>& img,
                                      const cv::Mat_<unsigned char>& mask) {
    assert(img.size() == mask.size());
    for (int y = 0; y < img.rows; ++y) {
        auto row = img.row(y);
        auto pm = mask.ptr<const unsigned char>(y);
        for (int x = 0; x < img.cols; ++x, ++pm)
            if (!*pm) row.col(x).setTo(T(0));
    }
}

template <typename T>
static void inplace_zero_outside_mask(cv::Mat_<T>& img, T zero = 0) {
    using namespace dlib_mesh;
    for (int i = 0; i < img.rows; ++i) {
        auto row = img.row(i);
        if (i >= mask_size ||
            mask[i][1] <= 0 || mask[i][0] >= img.cols)
            row.setTo(zero);
        else {
            if (mask[i][0] > 0)
                row.colRange(0, mask[i][0]).setTo(zero);
            const auto x = mask[i][0] + mask[i][1];
            if (x < img.cols)
                row.colRange(x, img.cols).setTo(zero);
        }
    }
}

template <typename T>
static cv::Mat_<T> row_from_mask(const cv::Mat_<T>& img) {
    using namespace dlib_mesh;
    assert(img.rows >= mask_size);
    auto row = cv::Mat_<T>(1, mask_size * img.cols);
    auto row_roi = cv::Rect(0,0,0,1);
    auto img_roi = cv::Rect(0,0,0,1);
    for (unsigned i = 0; i < mask_size; ++i) {
        auto& p = mask[i];
        if (p[1] > 0) {
            assert(p[0] + p[1] <= img.cols);
            img_roi.x = p[0];
            img_roi.width = p[1];
            row_roi.width = p[1];
            img(img_roi).copyTo(row(row_roi));
            row_roi.x += row_roi.width;
        }
        ++img_roi.y;
    }
    return row.colRange(0,row_roi.x);
}

namespace {
    struct lighting_matrix {
        const cv::Mat_<unsigned char> mean_img, mean_row;
        const cv::Mat_<signed char> eigenvectors;
        const double inner_norm;
        const double target_stddev;
        lighting_matrix(const cv::Mat_<unsigned char>& mean_img,
                        const cv::Mat_<unsigned char>& mean_row,
                        const cv::Mat_<signed char>& eigenvectors,
                        double inner_norm, double target_stddev)
            : mean_img(mean_img), mean_row(mean_row),
              eigenvectors(eigenvectors),
              inner_norm(inner_norm),
              target_stddev(target_stddev) {
        }
    };
}
static lighting_matrix get_lm(unsigned ver) {
    if (ver == 3)
        return { dlib_lm3::mean_img, dlib_lm3::mean_row, dlib_lm3::eigenvectors, dlib_lm3::inner_norm, dlib_lm3::target_stddev };
    throw std::invalid_argument("unknown lighting matrix");
}

raw_image::plane_ptr
static render_raw(const raw_image::plane& image,
                  const std::vector<cv::Point>& src_pts,
                  std::vector<cv::Point>& dest_pts,
                  const render_settings& rsettings,
                  const output_settings& osettings,
                  diagnostics* diag) {

    const auto iwidth = int(osettings.width);
    assert(iwidth >= 0);
    const auto iheight = int(osettings.height);
    assert(iheight >= 0);
    
    int minx = 0, maxx = iwidth;
    int miny = 0, maxy = iheight;
    for (const auto& p : dest_pts) {
        if (minx > p.x)
            minx = p.x;
        if (maxx <= p.x)
            maxx = p.x + 1;
        if (miny > p.y)
            miny = p.y;
        if (maxy <= p.y)
            maxy = p.y + 1;
    }
    assert(minx <= 0 && miny <= 0);
    assert(maxx >= iwidth && maxy >= iheight);

    // offset points
    for (auto& p : dest_pts)
        p.x -= minx, p.y -= miny;
    
    // src and dest image
    const auto src_img = to_Mat(image);
    const auto dest_img =
        cv::Mat(maxy-miny, maxx-minx, src_img.type(), cvScalarAll(128));
    const auto dest_rect = cv::Rect(-minx, -miny, iwidth, iheight);
    cv::Mat dest_roi = dest_img(dest_rect);
    
    // dest mesh
    const auto dest_mesh = cvx::compute_mesh(dest_pts);
    const auto dest_ordered = [&] {
        // order mesh so outer triangles are first
        std::vector<cvx::triangle_type> r(dest_mesh.begin(), dest_mesh.end());
        std::sort(r.begin(), r.end(),
                  [](const cvx::triangle_type& a, const cvx::triangle_type& b){
                      return a.back() > b.back();
                  });
        return r;
    }();

    // warp src to dest
    cvx::warp_mesh(dest_img, dest_pts, src_img, src_pts, dest_ordered);


    // lighting matrix
    const auto lm = get_lm(rsettings.lighting_matrix);
    auto lcomp = rsettings.lighting_compensation;
    assert(lm.eigenvectors.rows >= 0);
    if (lcomp > unsigned(lm.eigenvectors.rows))
        lcomp = unsigned(lm.eigenvectors.rows);
    
    // warp face to lighting mesh (inner only)
    const auto uc = cv::Mat(dlib_mesh::size, src_img.type(), cvScalarAll(128));
    cvx::warp_mesh(uc, dlib_mesh::pts,
                   src_img, src_pts.data(),
                   dlib_mesh::inner, dlib_mesh::inner + dlib_mesh::inner_size,
                   dlib_mesh::pts_size);
    cv::Mat_<unsigned char> u;
    if (uc.type() == CV_8UC1)
        u = uc;
    else if (same_channel_order(image.layout, raw_image::pixel::yuv)) {
        u = cv::Mat_<unsigned char>(dlib_mesh::size);
        static constexpr int from_to[] = {0,0};
        mixChannels(&uc, 1, &u, 1, from_to, 1);
    }
    else
        throw std::invalid_argument("color_space not supported");
    
    // mean and stdev
    cv::Scalar mean, stddev;
    meanStdDev(row_from_mask(u), mean, stddev);

    // standardize dest
    if (dest_roi.type() == CV_8UC1)
        addWeighted(dest_roi, lm.target_stddev/stddev[0], dest_roi, 0,
                    128 - mean[0]*lm.target_stddev/stddev[0],
                    dest_roi);
    else { // same_channel_order(color_space, raw_image::pixel::yuv)
        for (auto y = 0; y < dest_roi.rows; ++y) {
            auto px = dest_roi.ptr<unsigned char>(y);
            for (auto i = dest_roi.cols; i > 0; --i, px += 3) {
                const auto v = 128 + (*px-mean[0])*lm.target_stddev/stddev[0];
                *px = cv::saturate_cast<unsigned char>(v);
            }
        }
    }

    if (diag) {
        diag->lighting_weight = 0;
        diag->before_lighting = dest_roi;
    }

    auto r = raw_image::create(
        osettings.width, osettings.height, image.layout);

    if (lcomp > 0) {
        addWeighted(u, lm.target_stddev/stddev[0], u, 0,
                    128 - mean[0]*lm.target_stddev/stddev[0],
                    u);

        // u is face with mean 128 and target_stddev
    
        auto s = cv::Mat_<signed char>(dlib_mesh::size);
        subtract(u, lm.mean_img, s, cv::noArray(), s.type());

        // s = u - mean_face, mean 0

        inplace_zero_outside_mask(s);
        auto sr = s.reshape(0,1);

        // find eigenvector weights
        std::vector<double> weights;
        weights.reserve(lcomp);
        for (int i = 0; unsigned(i) < lcomp; ++i) {
            const auto x = lm.eigenvectors.row(i).dot(sr) / sqr(lm.inner_norm);
            weights.push_back(x);
        }

        if (diag)
            for (auto x : weights)
                diag->lighting_weight += float(sqr(x));
    
        // s <- estimated lighting (correction)
        sr.setTo(0);
        for (unsigned i = 0; i < weights.size(); ++i)
            addWeighted(sr, 1, lm.eigenvectors.row(int(i)), weights[i], 0, sr);

        // u = s + 128
        addWeighted(s, 1, s, 0, 128, u, u.type());

        // warp correction (from u)
        cv::Mat_<unsigned char> cu(dest_img.size(), 128);
        cvx::warp_mesh(cu, dest_pts.data(), u, dlib_mesh::pts,
                       dest_ordered.begin(), dest_ordered.end(),
                       dlib_mesh::pts_size);
        cv::Mat c;
        if (dest_img.type() == CV_8UC1)
            c = cu;
        else { // same_channel_order(color_space, raw_image::pixel::yuv)
            c = cv::Mat(dest_img.size(),dest_img.type(),cvScalarAll(128));
            static constexpr int from_to[] = {0,0};
            mixChannels(&cu, 1, &c, 1, from_to, 1);
        }

        // correction (subtract from dest_img)
        addWeighted(dest_roi, 1, c(dest_rect), -1, 128, to_Mat(*r));
    }
    
    else // lcomp == 0
        dest_roi.copyTo(to_Mat(*r));
    
    if (auto q = convert(*r, osettings.color_space))
        r = move(q);
    return r;
}

raw_image::plane_ptr
internal::render_dlib(const raw_image::plane& image,
                      const detected_coordinates& pos,
                      const render_settings& rsettings,
                      const output_settings& osettings,
                      diagnostics* diag) {

    // add predicted forehead and border points
    std::vector<cv::Point> pts;
    pts.reserve(dlib_mesh::pts_size);
    for (const auto& p : pos.landmarks)
        pts.push_back(raw_image::round_from(p));
    dlib_landmarks::inplace_push_border(pts);
    assert(pts.size() >= dlib_mesh::pts_size);

    // standardize points
    const auto std_pts = [&] {
        const auto eye_left  = midpoint(pts[36],pts[39]);
        const auto eye_right = midpoint(pts[42],pts[45]);
        const det::landmark_standardize<cv::Point2f> ls(eye_left, eye_right);
        std::vector<cv::Point2f> r;
        r.reserve(dlib_mesh::pts_inner);
        for (const auto& p : pts) {
            r.push_back(ls(p));
            if (r.size() >= dlib_mesh::pts_inner) break;
        }
        inplace_correct_pose(
            r, rsettings.pose_variant, rsettings.pose_compensation);
        return r;
    }();

    // dest points
    const auto xofs = osettings.width / 2.0;
    const auto yofs = double(osettings.eye_vertical) * osettings.height;
    std::vector<cv::Point> dest_pts;
    dest_pts.reserve(std_pts.size());
    for (const auto& p : std_pts) {
        const auto w = double(osettings.eye_width) * osettings.width;
        const auto x = cvRound(w * p.x + xofs);
        const auto y = cvRound(w * p.y + yofs);
        dest_pts.emplace_back(x,y);
    }
    dlib_landmarks::inplace_push_border(dest_pts);
    assert(dest_pts.size() >= dlib_mesh::pts_size);

    if (diag) {
        diag->final_landmarks = dest_pts;
        diag->final_landmarks.resize(68+3);
    }
    
    // convert source points (possibly mirrored)
    for (auto& p : pts)
        p = to_image_point(p,image);

    if (bytes_per_pixel(image.layout) == 1 ||
        (same_channel_order(image.layout, raw_image::pixel::yuv) &&
         bytes_per_pixel(osettings.color_space) > 1))
        return render_raw(image, pts, dest_pts, rsettings, osettings, diag);

    // crop image and convert to GRAY8 or YUV before render
    
    // find bounding box
    unsigned xlo = image.width, xhi = 0, ylo = image.height, yhi = 0;
    for (const auto& p : pts) {
        if (p.x < 0)
            xlo = 0;
        else {
            const auto x = unsigned(p.x);
            if (xlo > x)
                xlo = x;
            if (xhi <= x)
                xhi = x + 1;
        }
        if (p.y < 0)
            ylo = 0;
        else {
            const auto y = unsigned(p.y);
            if (ylo > y)
                ylo = y;
            if (yhi <= y)
                yhi = y + 1;
        }
    }
    if (xhi > image.width)
        xhi = image.width;
    if (yhi > image.height)
        yhi = image.height;
    if (xlo >= xhi || ylo >= yhi) {
        FILE_LOG(logWARNING) << "face to render not in image";
        return {};
    }

    // offset points
    for (auto& p : pts)
        p.x -= int(xlo), p.y -= int(ylo);
    
    // crop and convert
    const auto c = crop(image, xlo, ylo, xhi-xlo, yhi-ylo);
    const auto g = copy(
        c,
        (bytes_per_pixel(osettings.color_space) == 1 ||
         same_channel_order(osettings.color_space, raw_image::pixel::yuv)) ?
        osettings.color_space : raw_image::pixel::yuv);
    return render_raw(*g, pts, dest_pts, rsettings, osettings, diag);
}

void internal::in_place_correct_lighting_dlib(
    const raw_image::plane& image,
    const raw_image::plane& visibility,
    const detected_coordinates& pos,
    const render_settings& rsettings,
    diagnostics* diag) {
    
    if (!image.data || !visibility.data ||
        image.width != visibility.width ||
        image.height != visibility.height ||
        image.rotate || visibility.rotate ||
        image.scale || visibility.scale ||
        bytes_per_pixel(visibility.layout) != 1)
        throw std::invalid_argument("invalid image or visibility");

    if (pos.type != det::dt::dlib68 || pos.landmarks.size() != 68)
        throw std::invalid_argument("invalid landmarks (dlib68 required)");

    // lighting matrix
    const auto lm = get_lm(rsettings.lighting_matrix);
    auto lcomp = rsettings.lighting_compensation;
    assert(lm.eigenvectors.rows >= 0);
    if (lcomp > unsigned(lm.eigenvectors.rows))
        lcomp = unsigned(lm.eigenvectors.rows);

    // add predicted forehead and border points
    auto src_pts = [&]() {
        std::vector<cv::Point> pts;
        pts.reserve(dlib_mesh::pts_size);
        for (const auto& p : pos.landmarks)
            pts.push_back(raw_image::round_from(p));
        dlib_landmarks::inplace_push_border(pts);
        assert(pts.size() >= dlib_mesh::pts_size);
        for (auto& p : pts) {
            // make sure points are in image
            p.x = std::max(0,p.x);
            p.y = std::max(0,p.y);
            p.x = std::min(int(image.width-1),p.x);
            p.y = std::min(int(image.height-1),p.y);
        }
        return pts;
    }();

    // warp face to lighting mesh (inner only)
    auto src_img = to_Mat(image);
    const auto uc = cv::Mat(dlib_mesh::size, src_img.type(), cvScalarAll(128));
    cvx::warp_mesh(uc, dlib_mesh::pts, src_img, src_pts.data(),
                   dlib_mesh::inner, dlib_mesh::inner + dlib_mesh::inner_size,
                   dlib_mesh::pts_size);
    cv::Mat_<unsigned char> u;
    if (uc.type() == CV_8UC1)
        u = uc;
    else if (same_channel_order(image.layout, raw_image::pixel::yuv)) {
        u = cv::Mat_<unsigned char>(dlib_mesh::size);
        static constexpr int from_to[] = {0,0};
        mixChannels(&uc, 1, &u, 1, from_to, 1);
    }
    else
        throw std::invalid_argument("color_space not supported");

    if (diag)
        diag->before_lighting = u.clone();
    
    // warp visibility to lighting mesh (inner only)
    cv::Mat_<unsigned char> v(dlib_mesh::size);
    v = 0;
    cvx::warp_mesh(v, dlib_mesh::pts, to_Mat(visibility), src_pts.data(),
                   dlib_mesh::inner, dlib_mesh::inner + dlib_mesh::inner_size,
                   dlib_mesh::pts_size);
    inplace_zero_outside_mask(v);  // is this needed ?
    if (countNonZero(v) < 10000)
        throw std::runtime_error("face has insufficient visible pixels");
    
    // mean and stdev
    cv::Scalar mean, stddev;
    meanStdDev(u, mean, stddev, v);

    // standardize src
    if (src_img.type() == CV_8UC1)
        addWeighted(src_img, lm.target_stddev/stddev[0], src_img, 0,
                    128 - mean[0]*lm.target_stddev/stddev[0],
                    src_img);
    else { // same_channel_order(image.layout, raw_image::pixel::yuv)
        for (auto y = 0; y < src_img.rows; ++y) {
            auto px = src_img.ptr<unsigned char>(y);
            for (auto i = src_img.cols; i > 0; --i, px += 3) {
                const auto v = 128 + (*px-mean[0])*lm.target_stddev/stddev[0];
                *px = cv::saturate_cast<unsigned char>(v);
            }
        }
    }

    if (lcomp <= 0)
        return;  // brightness and contrast only

    addWeighted(u, lm.target_stddev/stddev[0], u, 0,
                128 - mean[0]*lm.target_stddev/stddev[0],
                u);

    // u is face with mean 128 and target_stddev
    
    auto s = cv::Mat_<signed char>(dlib_mesh::size);
    subtract(u, lm.mean_img, s, cv::noArray(), s.type());

    // s = u - mean_face, mean 0

    inplace_zero_outside_mask(s,v);
    auto sr = s.reshape(0,1);

    // find eigenvector weights
    std::vector<double> weights;
    weights.reserve(lcomp);
    for (int i = 0; unsigned(i) < lcomp; ++i) {
        const auto x = lm.eigenvectors.row(i).dot(sr) / sqr(lm.inner_norm);
        weights.push_back(x);
    }

    if (diag)
        for (auto x : weights)
            diag->lighting_weight += float(sqr(x));
    
    // s <- estimated lighting (correction)
    sr.setTo(0);
    for (unsigned i = 0; i < weights.size(); ++i)
        addWeighted(sr, 1, lm.eigenvectors.row(int(i)), weights[i], 0, sr);

    // u = s + 128
    addWeighted(s, 1, s, 0, 128, u, u.type());

    // src mesh
    const auto src_mesh = cvx::compute_mesh(src_pts);
    const auto src_ordered = [&] {
        // order mesh so outer triangles are first
        std::vector<cvx::triangle_type> r(src_mesh.begin(), src_mesh.end());
        std::sort(r.begin(), r.end(),
                  [](const cvx::triangle_type& a, const cvx::triangle_type& b){
                      return a.back() > b.back();
                  });
        return r;
    }();
    
    // warp correction (from u)
    cv::Mat_<unsigned char> cu(src_img.size(), 128);
    cvx::warp_mesh(cu, src_pts.data(), u, dlib_mesh::pts,
                   src_ordered.begin(), src_ordered.end(),
                   dlib_mesh::pts_size);
    cv::Mat c;
    if (src_img.type() == CV_8UC1)
        c = cu;
    else { // same_channel_order(image.layout, raw_image::pixel::yuv)
        c = cv::Mat(src_img.size(),src_img.type(),cvScalarAll(128));
        static constexpr int from_to[] = {0,0};
        mixChannels(&cu, 1, &c, 1, from_to, 1);
    }

    // correction (subtract from src_img)
    addWeighted(src_img, 1, c, -1, 128, src_img);
}
