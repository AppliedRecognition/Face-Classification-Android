
#include "scaled_chip.hpp"
#include "transform.hpp"
#include "linear_regression.hpp"

#include <cassert>

using namespace raw_image;

plane_ptr
raw_image::extract_image_chip(const multi_plane_arg& image,
                              const scaled_chip& chip,
                              pixel_layout layout) {
    const auto cx = chip.center.x + 0.5f;
    const auto cy = chip.center.y + 0.5f;
    const auto w = chip.width - 0.5f;
    const auto h = chip.height - 0.5f;
    const auto deg = float(chip.angle*(180/M_PI));
    return extract_region(image, cx, cy, w, h, deg,
                          chip.out_width, chip.out_height,
                          layout);
}

rotated_box
raw_image::retina_align(stdx::span<const point2f> pts,
                        float scale, float yofs) {

    std::array<point2f,5> buf;
    if (pts.size() == 68) {
        buf[0] = 0.25f * (pts[37] + pts[38] + pts[40] + pts[41]);
        buf[1] = 0.25f * (pts[43] + pts[44] + pts[46] + pts[47]);
        buf[2] = pts[30];
        buf[3] = pts[48];
        buf[4] = pts[54];
        pts = buf;
    }
    else if (pts.size() == 478) {
        buf[0] = pts[468];
        buf[1] = pts[473];
        buf[2] = pts[4];
        buf[3] = pts[61];
        buf[4] = pts[291];
        pts = buf;
    }
    else if (pts.size() == 6) { // BlazeFace has mouth_center
        pts = pts.subspan(0,4); // remove tragions
    }
    else if (pts.size() != 5)
        throw std::invalid_argument(
            "RetinaFace alignment requires eyes/nose/mouth landmarks");

    // parameters are: cos, sin, xofs, yofs
    // cos = scale * cos(angle)
    // sin = scale * sin(angle)
    // x' = x*cos - y*sin + xofs
    // y' = y*cos + x*sin + yofs
    // align to a 1x1 box centered at 0,0
    // so x in [-0.46,+0.46] and y in [-0.5,+0.5]
    linear_regression<float> reg(10);

    const auto y0 = yofs - 0.5f;  // eyes
    const auto y1 = yofs + 0.04f; // nose
    const auto y2 = yofs + 0.5f;  // mouth
    // left eye
    reg.add(pts[0].x, -0.46f,    -y0,  1.0f, 0.0f);
    reg.add(pts[0].y,     y0, -0.46f,  0.0f, 1.0f);

    // right eye
    reg.add(pts[1].x,  0.46f,    -y0,  1.0f, 0.0f);
    reg.add(pts[1].y,     y0,  0.46f,  0.0f, 1.0f);

    // nose tip
    reg.add(pts[2].x,   0.0f,    -y1,  1.0f, 0.0f);
    reg.add(pts[2].y,     y1,   0.0f,  0.0f, 1.0f);

    if (pts.size() == 4) {
        // mouth center
        reg.add(pts[3].x, 0.0f,  -y2,  1.0f, 0.0f);
        reg.add(pts[3].y,   y2, 0.0f,  0.0f, 1.0f);
    }
    else { // pts.size() == 5
        // left mouth corner
        reg.add(pts[3].x, -0.39f,    -y2,  1.0f, 0.0f);
        reg.add(pts[3].y,     y2, -0.39f,  0.0f, 1.0f);

        // right mouth corner
        reg.add(pts[4].x,  0.39f,    -y2,  1.0f, 0.0f);
        reg.add(pts[4].y,     y2,  0.39f,  0.0f, 1.0f);
    }

    auto c = reg.compute();
    assert(c.size() == 4);

    rotated_box rbox;
    rbox.center = { c[2], c[3] };
    rbox.angle = std::atan2(c[1], c[0]);
    rbox.width = rbox.height = scale * std::sqrt(c[0]*c[0] + c[1]*c[1]);
    return rbox;
}
