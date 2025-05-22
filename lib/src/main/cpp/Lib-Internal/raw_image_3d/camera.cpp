
#include "camera.hpp"

#include <raw_image/core.hpp>
#include <raw_image/pixels.hpp>

using namespace raw_image;

/*
  "example params": {
    "color": {
        "center": [322.80167,241.47034],
        "dcoeff": [0,0,0,...],
        "dmodel": "INVERSE_BROWN_CONRADY",
        "flen": [606.68915,605.2872],
        "width": 640
        "height": 480,
        "layout": "RGB8",
        "bytes_per_line": 1920,
        "bytes_per_pixel": 3,
    },
    "depth": {
        "center": [319.56256,239.35973],
        "dcoeff": [0,0,0,...],
        "dmodel": "BROWN_CONRADY",
        "flen": [593.32465,593.32465],
        "width": 640
        "height": 480,
        "scale": 0.001,
        "layout": "Z16",
        "bytes_per_line": 1280,
        "bytes_per_pixel": 2,
    },
    "translate": {
        "rotation": [0.9999945,0.0018742126,0.002727295,...],
        "translation": [-0.015015045,2.4198785e-05,-4.627119e-05]
    }
  }
*/

camera_registration::camera_registration(const json::object& params) {
    auto& color = get_object(params["color"]);
    auto& color_center = get_array(color["center"]);
    assert(color_center.size() == 2);
    this->color_center =
        { make_number(color_center[0]), make_number(color_center[1]) };
    if (!(0 < this->color_center.x) || !(0 < this->color_center.y))
        throw std::invalid_argument("invalid camera center");
    // todo: could also check for "width" and "height"

    auto& color_flen = get_array(color["flen"]);
    assert(color_flen.size() == 2);
    this->color_flen =
        { make_number(color_flen[0]), make_number(color_flen[1]) };
    if (!(0 < this->color_flen.x) || !(0 < this->color_flen.y))
        throw std::invalid_argument("invalid camera focal length");

    auto& depth = get_object(params["depth"]);
    auto& depth_center = get_array(depth["center"]);
    assert(depth_center.size() == 2);
    this->depth_center =
        { make_number(depth_center[0]), make_number(depth_center[1]) };
    if (!(0 < this->depth_center.x) || !(0 < this->depth_center.y))
        throw std::invalid_argument("invalid camera center");
    // todo: could also check for "width" and "height"

    auto& depth_flen = get_array(depth["flen"]);
    assert(depth_flen.size() == 2);
    this->depth_flen =
        { make_number(depth_flen[0]), make_number(depth_flen[1]) };
    if (!(0 < this->depth_flen.x) || !(0 < this->depth_flen.y))
        throw std::invalid_argument("invalid camera focal length");

    auto& trans = get_object(params["translate"]);
    auto& ta = get_array(trans["translation"]);
    assert(ta.size() == 3);
    this->translate =
        { make_number(ta[0]), make_number(ta[1]), make_number(ta[2]) };
    if (length_squared(this->translate) < 1)
        this->translate *= 1000; // convert metres -> mm

    auto& ra = get_array(trans["rotation"]);
    assert(ra.size() == 9);
    this->rotate =
        { point3f { make_number(ra[0]), make_number(ra[1]), make_number(ra[2]) },
          point3f { make_number(ra[3]), make_number(ra[4]), make_number(ra[5]) },
          point3f { make_number(ra[6]), make_number(ra[7]), make_number(ra[8]) } };
    //this->rotate = transpose(this->rotate);
}

bool camera_registration::is_identity() const {
    if (length_squared(color_center - depth_center) > 0.125f)
        return false;
    if (length_squared(color_flen - depth_flen) > 0.125f)
        return false;
    if (length_squared(translate) > 0.125f)
        return false;
    const auto diff = rotate - I3x3f;
    for (auto& row : diff.rows)
        if (length_squared(row) > 1e-5)
            return false;
    return true;
}

point_result
camera_registration::operator()(
    point2i depth_pixel, unsigned depth_value) const {
    if (depth_value <= 0)
        throw std::invalid_argument(
            "for camera_registration, depth value must be non-zero");
    point_result r;
    r.dpx = depth_pixel;
    const auto dv = float(depth_value);
    r.dloc = {
        dv * (float(r.dpx.x) - depth_center.x) / depth_flen.x,
        dv * (float(r.dpx.y) - depth_center.y) / depth_flen.y,
        dv
    };
    r.cloc = translate + rotate * r.dloc;
    const auto x = color_center.x + color_flen.x * r.cloc.x / r.cloc.z;
    const auto y = color_center.y + color_flen.y * r.cloc.y / r.cloc.z;
    r.cpx = { stdx::round_from(x), stdx::round_from(y) };
    return r;
}

void camera_registration::align(const raw_image::plane& depth_src,
                                const raw_image::plane& depth_dest,
                                unsigned max_depth) const {

    memset(depth_dest.data, 0, depth_dest.height * depth_dest.bytes_per_line);

    raw_image::pixels<uint16_t> dest(depth_dest);
    const raw_image::pixels<uint16_t> src(depth_src);
    float dy = -depth_center.y;
    for (auto&& line : src) {
        const float y = dy / depth_flen.y;
        dy += 1;
        float dx = -depth_center.x;
        for (const auto z : line) {
            if (z < 128 || max_depth < z) {
                dx += 1;
                continue; // missing or out of range
            }

            // real world location relative to depth camera
            const auto dloc =
                point3f { z * dx / depth_flen.x, z * y, float(z) };
            dx += 1;

            // real world location relative to color camera
            // todo_check_this;
            const auto cloc = point3f(translate + rotate * dloc);
            //const auto cloc = point3f(rotate * (dloc + translate));

            const auto zi = stdx::round_to<int>(std::floor(cloc.z));
            if (zi < 64) continue; // this shouldn't happen

            // pixel on depth_dest (color) image
            const auto cx = stdx::round_to<int>(
                color_center.x + color_flen.x * cloc.x / cloc.z);
            if (cx < 0 || depth_dest.width <= unsigned(cx))
                continue;
            const auto cy = stdx::round_to<int>(
                color_center.y + color_flen.y * cloc.y / cloc.z);
            if (cy < 0 || depth_dest.height <= unsigned(cy))
                continue;

            // if multiple src pixels map to same dest pixel, store min value
            auto& px = dest(cx,cy);
            if (px <= 0 || zi < px)
                px = uint16_t(zi);
        }
    }
}

raw_image::plane_ptr
camera_registration::align(
    const raw_image::plane& depth_src, unsigned max_depth) const {
    auto dest = create(depth_src.width, depth_src.height, depth_src.layout);
    align(depth_src, *dest, max_depth);
    return dest;
}

/*
std::vector<rgbxyz>
camera_registration::merge(const raw_image::plane& depth,
                           const raw_image::plane& color,
                           unsigned max_depth) const {

    // assuming rgba has alpha in msb (ie. little-endian)
    // and default alpha is 0
    const auto rgba = copy(color, raw_image::pixel::rgba32);
    raw_image::pixels<uint32_t> cimg(rgba);

    static constexpr auto u24 = 1u << 24;
    if (0) { // test assumptions
        for (auto&& line : cimg)
            for (auto i : line)
                assert(i < u24);
    }

    std::vector<rgbxyz> vec;
    vec.reserve(std::min(depth.width*depth.height, color.width*color.height));

    const raw_image::pixels<uint16_t> dimg(depth);
    float dy = -depth_center.y;
    for (auto&& dline : dimg) {
        const float y = dy / depth_flen.y;
        dy += 1;
        float dx = -depth_center.x;
        for (const auto z : dline) {
            if (z < 128 || max_depth < z) {
                dx += 1;
                continue; // missing or out of range
            }

            // location relative to depth camera
            const auto dloc =
                point3f { z * dx / depth_flen.x, z * y, float(z) };
            dx += 1;

            // location relative to color camera
            const auto cloc = point3f(translate + rotate * dloc);
            const auto xi = (1<<12) + stdx::round_to<int>(4*cloc.x);
            if (xi < 0 || (1<<13) <= xi) continue;
            const auto yi = (1<<12) + stdx::round_to<int>(4*cloc.y);
            if (yi < 0 || (1<<13) <= yi) continue;
            const auto zi = stdx::round_to<int>(4*cloc.z);
            if ((1<<14) <= zi) continue;

            // pixel on color image
            const auto cx = stdx::round_to<int>(
                color_center.x + color_flen.x * cloc.x / cloc.z);
            if (cx < 0 || color.width <= unsigned(cx))
                continue;
            const auto cy = stdx::round_to<int>(
                color_center.y + color_flen.y * cloc.y / cloc.z);
            if (cy < 0 || color.height <= unsigned(cy))
                continue;

            auto& cpx = cimg(cx,cy);
            if (cpx < u24) {
                const auto idx = u24 + unsigned(vec.size());
                vec.push_back( {
                        cpx,          // red
                        cpx >> 8,     // green
                        cpx >> 16,    // blue
                        unsigned(xi),
                        unsigned(yi),
                        unsigned(zi)
                    } );
                cpx = idx;
            }
            else { // this pixel already used
                const auto idx = cpx - u24;
                assert(idx < vec.size());
                auto& other = vec[idx];
                if (zi < other.z) {
                    // it's actually closer than previously recorded
                    other.x = xi;
                    other.y = yi;
                    other.z = zi;
                }
            }
        }
    }

    return vec;
}
*/


/****************  NEW METHODS  ****************/

/*  "calib": {
        "Distc": [-0.113,-0.152,-0.004,-0.004,0.097],
        "Invdistc": [0.112,0.173,0.004,0.004,-0.0759],
        "Kc": [[1.48,0,-0.02001],[0,1.975,0.0224],[0,0,1]],

        "Distp": [0.002,-0.010,-0.011,-0.001,-0.004],
        "Invdistp": [-0.0023,0.01073,0.01149,0.001,0.00472],
        "Kp": [[1.893,0,-0.000261],[0,1.363,0.00575],[0,0,1]],
        "Pp": [[-0.000167,1.893,-0.0092,-0.0011],
               [1.356,-0.00053,-0.1334,64.708],
               [0.102,0.00475,0.9947,4.51865]],
        "Rp": [[-0.001,0.99998,-0.0047],
               [0.9947,-0.00041,-0.1020],
               [0.1020,0.0047,0.99476]],
        "Tp": [0,47.4522,4.518],

        "Distt": [0,0,0,0,0],
        "Invdistt": [0,0,0,0,0],
        "Kt": [[1.449,0,-0.025768],[0,2.577,0.04388],[0,0,1]],
        "Pt": [[1.4497,0.007141,-0.0271,37.1569],
               [-0.0123,2.577,0.058381,-1.71],
               [0.00097,-0.00561,0.99998,3.939]],
        "Rt": [[0.99998,0.00482,-0.000952],
               [-0.0048,0.999972,0.005624],
               [0.000979,-0.00561,0.99998]],
        "Tt": [25.6999,-0.7332,3.939]

        "QV": [0.0185,0.03984,-0.01238,-12.7052,-29.553,1009.675],
        "Rmax": 8191,
    },
    "color": [1920,1080],
    "depth": [640,480],
    "ver": 0
 */
static void unpack(const json::value& v, point3f& dest) {
    const auto arr = json::make_array<float,3>(v);
    dest = { arr[0], arr[1], arr[2] };
}
static void unpack(const json::value& v, matrix3x3f& dest) {
    auto& arr = get_array(v);
    assert(arr.size() == 3);
    for (unsigned i = 0; i < 3; ++i)
        unpack(arr[i], dest.rows[i]);
}
template <typename T>
static inline T unpack(const json::value& v) {
    T dest; unpack(v, dest); return dest;
}

metadata::metadata(const json::object& obj, const sr300_t&) {
    if (obj["ver"] != json::integer(0))
        throw std::runtime_error("incorrect SR300 parameter version");
    auto& calib = get_object(obj["calib"]);

    const auto ddims = json::make_array<int,2>(obj["depth"]);
    if (ddims[0] <= 0 || ddims[1] <= 0)
        throw std::runtime_error("invalid SR300 color dimensions");
    depth.width = ddims[0], depth.height = ddims[1];

    const auto cdims = json::make_array<int,2>(obj["color"]);
    if (cdims[0] <= 0 || cdims[1] <= 0)
        throw std::runtime_error("invalid SR300 color dimensions");
    color.width = cdims[0], color.height = cdims[1];

    const auto Kcm = unpack<matrix3x3f>(calib["Kc"]);
    auto& Kc = Kcm.rows;
    depth.center = {
        (Kc[0][2] * 0.5f + 0.5f) * float(depth.width),
        (Kc[1][2] * 0.5f + 0.5f) * float(depth.height),
    };
    depth.flen = {
        Kc[0][0] * 0.5f * float(depth.width),
        Kc[1][1] * 0.5f * float(depth.height),
    };
    depth.model = distortion::inverse_brown_conrady;
    depth.coeffs = json::make_array<float,5>(calib["Invdistc"]);

    const auto Ktm = unpack<matrix3x3f>(calib["Kt"]);
    auto& Kt = Ktm.rows;
    color.center = {
        Kt[0][2] * 0.5f + 0.5f,
        Kt[1][2] * 0.5f + 0.5f,
    };
    color.flen = {
        Kt[0][0] * 0.5f * float(color.width),
        Kt[1][1] * 0.5f * float(color.height),
    };
    if (color.width*3 == color.height*4) { // adjust if 4:3 aspect ratio
        color.flen.x *= 4.0f / 3;
        color.center.x *= 4.0f / 3;
        color.center.x -= 1.0f / 6;
    }
    color.center.x *= float(color.width);
    color.center.y *= float(color.height);

    translate.rotation = unpack<matrix3x3f>(calib["Rt"]);
    translate.translation = unpack<point3f>(calib["Tt"]);
}

point3f intrinsics::deproject(point2i pixel, float depth) const {
    float x = (float(pixel.x) - center.x) / flen.x;
    float y = (float(pixel.y) - center.y) / flen.y;

    switch (model) {
    case distortion::inverse_brown_conrady: {
        // need to loop until convergence
        // 10 iterations determined empirically
        float xo = x, yo = y;
        for (int i = 0; i < 10; ++i) {
            float r2 = x * x + y * y;
            float icdist = 1.0f / float(1 + ((coeffs[4] * r2 + coeffs[1]) * r2 + coeffs[0]) * r2);
            float xq = x / icdist;
            float yq = y / icdist;
            float delta_x =
                2 * coeffs[2] * xq * yq +
                coeffs[3] * (r2 + 2 * xq * xq);
            float delta_y =
                2 * coeffs[3] * xq * yq +
                coeffs[2] * (r2 + 2 * yq * yq);
            x = (xo - delta_x) * icdist;
            y = (yo - delta_y) * icdist;
        }
        break;
    }

    case distortion::none:
        break;

    default:
        throw std::runtime_error("invalid distortion model");
    }
    return { x * depth, y * depth, depth };
}

std::vector<point3f_point2h>
metadata::map_depth(const raw_image::plane& dimg, int rot_var) const {
    if (depth.width <= 0 || dimg.width != unsigned(depth.width) ||
        depth.height <= 0 || dimg.height != unsigned(depth.height))
        throw std::invalid_argument("depth image dimensions don't match metadata");

    auto rot = translate.rotation;
    if (rot_var < 0)
        rot = transpose(rot);
    else if (rot_var == 0)
        rot = raw_image::I3x3f;
    auto& trans = translate.translation;

    std::vector<point3f_point2h> vec;
    vec.reserve(unsigned(depth.height) * std::size_t(depth.width));
    point2i dpx = {0,0};
    for (auto&& row : raw_image::pixels<uint16_t>(dimg)) {
        dpx.x = 0;
        for (auto z : row) {
            ++dpx.x;
            auto& r = vec.emplace_back();
            if (128 <= z && z < 8192) {
                const auto dpt = depth.deproject(dpx, z);
                const auto cpt = rot*dpt + trans;
                const auto cpx = color.project(cpt);
                const auto x = std::lround(cpx.x);
                const auto y = std::lround(cpx.y);
                if (0 <= x && x < color.width &&
                    0 <= y && y < color.height) {
                    r.real = dpt;
                    r.color = { int16_t(x), int16_t(y) };
                }
            }
        }
        ++dpx.y;
    }
    return vec;
}
