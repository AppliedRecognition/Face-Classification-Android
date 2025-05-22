
#include "dlib_landmarks.hpp"
#include "opencv_operators.hpp"
#include <det/landmark_standardize.hpp>
#include <raw_image/point_rounding.hpp>

static constexpr auto forehead_predict_pts =
    std::array<unsigned,4>{0u,16u,30u,33u};

using point_type = cv::Point2f;
static const point_type forehead_predict_0x[] = {
    {-0.0454921f, -0.307841f},
    {-0.120475f,   0.276975f},
    {-0.0893545f, -0.00181544f},
    {-0.313703f,  -0.0588676f},
};
static const point_type forehead_predict_0y[] = {
    { 0.121325f,   0.0238361f},
    {-0.0682846f, -0.0640516f},
    { 0.0375445f, -0.00360724f},
    { 0.0118831f, -0.164269f},
};

static const point_type forehead_predict_1x[] = {
    {-0.0963294f, -0.32942f},
    {-0.0963294f,  0.32942f},
    {-0.114984f,   0},
    {-0.316119f,   0},
};
static const point_type forehead_predict_1y[] = {
    { 0.109349f,  -0.0274274f},
    {-0.109349f,  -0.0274274f},
    { 0,           0.0140924f},
    { 0,          -0.187702f},
};

static const point_type forehead_predict_2x[] = {
    {-0.120475f,  -0.276975f},
    {-0.0454922f,  0.307841f},
    {-0.0893546f,  0.00181513f},
    {-0.313703f,   0.0588679f},
};
static const point_type forehead_predict_2y[] = {
    { 0.0682846f, -0.0640516f},
    {-0.121325f,   0.0238361f},
    {-0.0375442f, -0.00360727f},
    {-0.0118834f, -0.164269f},
};

using landmark_standardize = det::landmark_standardize<cv::Point2f>;

static void inplace_push_forehead_raw(std::vector<cv::Point>& pts,
                                      const landmark_standardize& ls) {
    using namespace dlib_landmarks;
    unsigned j = 0;
    auto v0 = mean[68];
    auto v1 = mean[69];
    auto v2 = mean[70];
    for (auto i : forehead_predict_pts) {
        const auto p = ls(pts[i]) - mean[i];
        v0.x += p.dot(forehead_predict_0x[j]);
        v0.y += p.dot(forehead_predict_0y[j]);
        v1.x += p.dot(forehead_predict_1x[j]);
        v1.y += p.dot(forehead_predict_1y[j]);
        v2.x += p.dot(forehead_predict_2x[j]);
        v2.y += p.dot(forehead_predict_2y[j]);
        ++j;
    }
    pts.push_back(raw_image::round_from(ls.recover(v0)));
    pts.push_back(raw_image::round_from(ls.recover(v1)));
    pts.push_back(raw_image::round_from(ls.recover(v2)));
}

void dlib_landmarks::inplace_push_forehead(std::vector<cv::Point>& pts) {
    assert(pts.size() == 68);
    const auto eye_left  = midpoint(pts[36],pts[39]);
    const auto eye_right = midpoint(pts[42],pts[45]);
    const landmark_standardize ls(eye_left, eye_right);
    pts.reserve(68 + 3);
    inplace_push_forehead_raw(pts, ls);
}

void dlib_landmarks::inplace_push_border(std::vector<cv::Point>& pts) {
    assert(pts.size() >= 68);
    const auto eye_left  = midpoint(pts[36],pts[39]);
    const auto eye_right = midpoint(pts[42],pts[45]);
    const landmark_standardize ls(eye_left, eye_right);

    if (pts.size() == 68) {
        pts.reserve(68 + 3 + 12);
        inplace_push_forehead_raw(pts, ls);
    }
    assert(pts.size() == 71);

    float left = 0, top = 0, right = 0, bottom = 0;
    for (unsigned i = 0; i <= 16; ++i) {
        const auto p = ls(pts[i]);
        if (left   > p.x) left   = p.x;
        if (right  < p.x) right  = p.x;
        if (top    > p.y) top    = p.y;
        if (bottom < p.y) bottom = p.y;
    }
    for (unsigned i = 68; i < 71; ++i) {
        const auto p = ls(pts[i]);
        if (left   > p.x) left   = p.x;
        if (right  < p.x) right  = p.x;
        if (top    > p.y) top    = p.y;
        if (bottom < p.y) bottom = p.y;
    }
    assert(left < 0 && top < 0 && right > 0 && bottom > 0);

    const auto dx = right - left;
    const auto dy = bottom - top;
    left -= dx / 8;
    top  -= dy / 8;
    const auto w = dx * 1.25f;
    const auto h = dy * 1.25f;
    
    pts.reserve(71 + 12);

    pts.push_back(raw_image::round_from(ls.recover(left, top)));
    pts.push_back(raw_image::round_from(ls.recover(left, top+h/3)));
    pts.push_back(raw_image::round_from(ls.recover(left, top+h*2/3)));
    pts.push_back(raw_image::round_from(ls.recover(left, top+h)));

    pts.push_back(raw_image::round_from(ls.recover(left+w/3,   top+h)));
    pts.push_back(raw_image::round_from(ls.recover(left+w*2/3, top+h)));

    pts.push_back(raw_image::round_from(ls.recover(left+w, top+h)));
    pts.push_back(raw_image::round_from(ls.recover(left+w, top+h*2/3)));
    pts.push_back(raw_image::round_from(ls.recover(left+w, top+h/3)));
    pts.push_back(raw_image::round_from(ls.recover(left+w, top)));

    pts.push_back(raw_image::round_from(ls.recover(left+w*2/3, top)));
    pts.push_back(raw_image::round_from(ls.recover(left+w/3,   top)));
}

