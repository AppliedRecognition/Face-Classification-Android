
#include "image.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal_image.hpp"

#include <raw_image_io/io.hpp>
#include <raw_image/transform.hpp>

#include <applog/core.hpp>


using namespace det;


static inline
bool operator==(const raw_image::image_size& a,
                const raw_image::image_size& b) {
    return a.width == b.width && a.height == b.height;
}
static inline bool is_gray(const raw_image::multi_plane_arg& img) {
    return img.size() == 1 && bytes_per_pixel(img.front().layout) == 1;
}
static inline bool is_color(const raw_image::multi_plane_arg& img) {
    return !is_gray(img);
}

static auto find_any_gray8(const raw_image::multi_plane_arg& img) {
    const raw_image::plane* r = nullptr;
    for (auto& plane : img)
        if (same_channel_order(raw_image::pixel::gray8, plane.layout))
            r = &plane;
    return r;
}
static inline auto
gray8_or_empty(const raw_image::multi_plane_arg& img) {
    if (auto p = find_any_gray8(img))
        return *p;
    return raw_image::plane{};
}
/*
static auto find_r85g10b05(const raw_image::multi_plane_arg& img) {
    const raw_image::plane* r = nullptr;
    for (auto& plane : img)
        if (plane.layout == raw_image::pixel::r85g10b05)
            r = &plane;
    return r;

}
*/

template <typename T>
image_struct::image_struct(const T& img, std::enable_if_t<std::is_same_v<T,raw_image::multi_plane_arg> >*)
    : color(img),
      gray(gray8_or_empty(img)),
      size(dimensions(color.multiplane)) {
}

image_struct::image_struct(const raw_image::plane& img)
    : color(img),
      gray(img),
      size(dimensions(color.multiplane)) {
}

image_struct::image_struct(std::unique_ptr<raw_image::plane> img)
    : color(move(img)),
      gray(color.multiplane.front()),
      size(dimensions(color.multiplane)) {
}

image_struct::image_struct(std::shared_ptr<const raw_image::plane> img)
    : color(move(img)),
      gray(color.multiplane.front()),
      size(dimensions(color.multiplane)) {
}

template <typename C, typename G>
image_struct::image_struct(C&& color, G&& gray)
    : color(std::move(color)),
      gray(std::move(gray)),
      size(dimensions(this->color.multiplane)) {
    assert(is_color(this->color.multiplane) && is_gray(this->gray.plane));
}

void image_deleter::operator()(const image_struct* p) {
    delete p;
}

static inline bool prefer_gray(const detection_settings& settings) {
    return settings.detector_version < 4;
}
static inline bool prefer_rgb24(const raw_image::plane& img,
                                const detection_settings& settings) {
    return !prefer_gray(settings) &&
        img.layout != raw_image::pixel::rgb24 &&
        img.layout != raw_image::pixel::bgr24;
}

static bool require_rotate(const raw_image::plane& img,
                           const detection_settings& settings) {
    return (img.rotate&3) != 0 && settings.detector_version > 0;
}

/// actually for v4 and later
static inline auto color_for_v4(raw_image::pixel_layout cs) {
    return cs == raw_image::pixel::bgr24 ? cs : raw_image::pixel::rgb24;
}
static inline auto
color_for_any(const stdx::span<const raw_image::plane>& cimg) {
    for (auto& plane : cimg) {
        const auto cc = to_color_class(plane.layout);
        if (cc == raw_image::cc::yuv_nv21)
            return raw_image::pixel::yuv24_nv21;
        if (cc == raw_image::cc::yuv_jpeg)
            return raw_image::pixel::yuv24_jpeg;
        if (plane.layout == raw_image::pixel::bgr24)
            return raw_image::pixel::bgr24;
    }
    return raw_image::pixel::rgb24;
}
static inline auto
gray_for_any(const stdx::span<const raw_image::plane>& cimg) {
    if (is_gray(cimg))
        return cimg.front().layout;
    for (auto& plane : cimg)
        if (to_color_class(plane.layout) == raw_image::cc::yuv_nv21)
            return raw_image::pixel::y8_nv21;
    return raw_image::pixel::gray8;
}

static inline auto pixels_for_detection(const detection_settings& settings) {
    if (settings.detector_version <= 3)
        return 340 * 1000 * settings.size_range;
    else
        return 500 * 1000 * settings.size_range;
}
static auto copy_for_detection(
    const stdx::span<const raw_image::plane>& cimg,
    const detection_settings& settings,
    raw_image::pixel_layout cs) {
    assert(!cimg.empty());
    const auto rot = (cimg.front().rotate&3) ? cimg.front().rotate : 0;
    if (cimg.size() == 2 &&
        cimg.front().width == 2*cimg.back().width &&
        cimg.front().height == 2*cimg.back().height) {
        const auto min_px = pixels_for_detection(settings);
        if (float(cimg.back().width) * float(cimg.back().height) >= min_px) {
            // downscale what we assume is the Y plane of a YUV image
            FILE_LOG(logDETAIL) << "scaling y-plane from "
                                << cimg.front().width << 'x'
                                << cimg.front().height << " to "
                                << cimg.back().width << 'x'
                                << cimg.back().height;
            const auto y_plane = copy_resize(
                cimg.front(), cimg.back().width, cimg.back().height);
            y_plane->scale = cimg.front().scale + 1;
            const raw_image::plane img[] = { *y_plane, cimg.back() };
            return copy_rotate(img, rot, cs);
        }
    }
    return copy_rotate(cimg, rot, cs);
}
static inline auto copy_gray(
    stdx::span<const raw_image::plane> cimg,
    const detection_settings& settings) {
    const auto cs = gray_for_any(cimg); // todo: raw_image::pixel::r85g10b05
    return copy_for_detection(cimg, settings, cs);
}
static inline auto copy_color(
    const stdx::span<const raw_image::plane>& cimg,
    const detection_settings& settings) {
    if (is_gray(cimg))
        return copy_gray(cimg, settings);
    const auto cs = settings.detector_version >= 4 ?
        color_for_v4(cimg.front().layout) : color_for_any(cimg);
    return copy_for_detection(cimg, settings, cs);
}
static inline auto copy_preferred(
    const stdx::span<const raw_image::plane>& cimg,
    const detection_settings& settings) {
    return settings.detector_version >= 4 ?
        copy_color(cimg, settings) : copy_gray(cimg, settings);
}


image_type internal::copy_image(
    core::active_job,
    const detection_settings& settings, 
    const raw_image::multi_plane_arg& cimg,
    const stdx::options_tuple<gray_option,color_option>& opts) {

    if (cimg.empty())
        throw std::invalid_argument("image is empty");

    const auto dv = settings.detector_version;
    const auto g = bool(std::get<gray_option>(opts));
    const auto c = bool(std::get<color_option>(opts));

    raw_image::plane_ptr color, gray;
    if (dv > 0) {
        // create image for face detection
        if (c == g) {
            gray = copy_preferred(cimg, settings);
            if (bytes_per_pixel(gray) > 1)
                swap(color, gray);
        }
        else if (g || is_gray(cimg))
            gray = copy_gray(cimg, settings);
        else // c
            color = copy_color(cimg, settings);
    }

    // create other image if necessary
    if (!color && c && is_color(cimg))
        color = copy(cimg, color_for_any(cimg));
    if (!gray && (g || !color))
        gray = copy(cimg, gray_for_any(cimg));

    if (!color)
        return image_type(new image_struct(move(gray)));
    if (!gray)
        return image_type(new image_struct(move(color)));
    return image_type(new image_struct(move(color), move(gray)));
}

image_type internal::take_image(
    core::active_job,
    const detection_settings& settings, 
    std::unique_ptr<raw_image::plane> img,
    const stdx::options_tuple<gray_option,color_option>& opts) {

    if (!img)
        throw std::invalid_argument("image is nullptr");
    if (!manages_pixel_buffer(img))  // todo: should this be an exception?
        FILE_LOG(logWARNING) << "move_image: image pixel buffer is not included in unique_ptr";

    const auto g = bool(std::get<gray_option>(opts));
    const auto c = bool(std::get<color_option>(opts));

    if (require_rotate(*img, settings))
        in_place_rotate(*img);

    if (g && !c) {
        auto p = convert(*img, raw_image::pixel::gray8);
        assert(!p); // convert must be in place
    }
    else if (prefer_rgb24(*img, settings) && is_color(*img)) {
        auto p = convert(*img, raw_image::pixel::rgb24);
        assert(!p); // convert must be in place
    }
    
    if (g && c && is_color(*img)) {
        auto gray = copy(img, raw_image::pixel::gray8);
        return image_type(new image_struct(move(img), move(gray)));
    }
    return image_type(new image_struct(move(img)));
}

image_type internal::share_image(
    core::active_job context,
    const detection_settings& settings, 
    std::shared_ptr<const raw_image::plane> cimg,
    const stdx::options_tuple<gray_option,color_option>& opts) {
    
    if (!cimg)
        throw std::invalid_argument("image is nullptr");
    if (!manages_pixel_buffer(cimg))  // todo: should this be an exception?
        FILE_LOG(logWARNING) << "use_image: image pixel buffer is not included in unique_ptr";

    const auto g = bool(std::get<gray_option>(opts));
    const auto c = bool(std::get<color_option>(opts));

    if (require_rotate(*cimg, settings) || (g && !c && is_color(*cimg)))
        return copy_image(std::move(context), settings, cimg, opts);

    if (g && c && is_color(*cimg)) {
        auto gray = copy(cimg, raw_image::pixel::gray8);
        return image_type(new image_struct(move(cimg), move(gray)));
    }
    return image_type(new image_struct(move(cimg)));
}

image_type internal::share_pixels(
    core::active_job,
    const detection_settings& settings, 
    const raw_image::multi_plane_arg& cimg,
    const stdx::options_tuple<gray_option,color_option>& opts) {

    if (cimg.empty())
        throw std::invalid_argument("image is empty");

    const auto dv = settings.detector_version;
    const auto rot = (cimg.front().rotate&3) ? cimg.front().rotate : 0;

    if (is_gray(cimg)) {
        // gray only
        if (dv == 0 || !rot)
            return image_type(new image_struct(cimg));
        FILE_LOG(logDETAIL) << "use_image: copying gray image";
        return image_type(new image_struct(copy_gray(cimg, settings)));
    }

    // image is color (or multi-plane)

    auto g = bool(std::get<gray_option>(opts));
    auto c = bool(std::get<color_option>(opts));

    const auto gp = find_any_gray8(cimg);

    if (!g && !c) {
        // determine whether we want gray or color
        if (dv >= 4)
            c = true;
        else if ((dv > 0 && rot) || gp || cimg.size() != 1)
            g = true;
        else
            c = true;
    }
    assert(g || c);

    raw_image::plane_ptr color, gray;
    if (dv > 0) {
        if (c && (dv >= 4 || !g)) {
            // face detection will use color image
            if (rot || cimg.size() != 1) {
                FILE_LOG(logDETAIL) << "use_image: copying color image";
                color = copy_color(cimg, settings);
            }
        }
        else {
            // face detection will use gray image
            if (rot || !gp) {
                FILE_LOG(logDETAIL) << "use_image: copying gray image";
                gray = copy_gray(cimg, settings);
            }
        }
    }

    if (g && !gray && !gp) {
        FILE_LOG(logDETAIL) << "use_image: copying gray image";
        gray = copy(cimg, gray_for_any(cimg));
    }
    if (c && !color && cimg.size() != 1) {
        FILE_LOG(logDETAIL) << "use_image: copying color image";
        color = copy(cimg, color_for_any(cimg));
    }

    if (!g) {
        if (color)
            return image_type(new image_struct(move(color)));
        else
            return image_type(new image_struct(cimg));
    }
    if (!c) {
        if (gray)
            return image_type(new image_struct(move(gray)));
        else
            return image_type(new image_struct(*gp));
    }

    // color and gray
    if (!gray) {
        if (color)
            return image_type(new image_struct(move(color), *gp));
        else
            return image_type(new image_struct(cimg, *gp));
    }
    if (!color)
        return image_type(new image_struct(cimg, move(gray)));
    return image_type(new image_struct(move(color), move(gray)));
}

image_type internal::use_pixels(
    core::active_job,
    const detection_settings& settings, 
    const raw_image::plane& cimg,
    const stdx::options_tuple<gray_option,color_option>& opts) {

    const auto g = bool(std::get<gray_option>(opts));
    const auto c = bool(std::get<color_option>(opts));

    auto img = cimg;
    if (require_rotate(img, settings))
        in_place_rotate(img);
    
    if (g && !c) {
        auto p = convert(img, raw_image::pixel::gray8);
        assert(!p); // convert must be in place
    }
    else if (prefer_rgb24(img, settings)) {
        auto p = convert(img, raw_image::pixel::rgb24);
        assert(!p); // convert must be in place
    }
    
    if (g && c && is_color(img)) {
        auto gray = copy(img, raw_image::pixel::gray8);
        return image_type(new image_struct(img, move(gray)));
    }
    return image_type(new image_struct(img));
}

unsigned det::suggested_scaling(const detection_settings& s, 
                                const raw_image::image_size& size) {
    const auto desired_pix = [&]() -> float {
        if (s.landmark_detection.landmarks != lm::none)
            return 1000;
        else if (s.detector_version >= 4)
            return 500;
        else
            return 340;
    }() * 1000 * s.size_range;
    if (desired_pix < 10) {
        FILE_LOG(logWARNING) << "detection.size_range too small";
        return 8;
    }
    const auto scale = float(size.width) * float(size.height) / desired_pix;
    if (scale >= 8*8)
        return 8;
    if (scale >= 4*4)
        return 4;
    if (scale >= 2*2)
        return 2;
    return 1;
}

raw_image::image_size
det::get_image_dimensions(stdx::arg<const image_struct> image) {
    if (!image) {
        FILE_LOG(logERROR) << "get_image_dimensions: invalid image";
        throw std::invalid_argument("invalid image argument");
    }
    return image->size;
}

const raw_image::plane&
internal::get_raw_from_image(
    stdx::arg<const image_struct> image,
    const stdx::options_tuple<gray_option,color_option>& opts) {
    if (!image) {
        FILE_LOG(logERROR) << "get_raw_from_image: invalid image";
        throw std::invalid_argument("invalid image argument");
    }
    if (image->gray.plane.data == nullptr) {
        if (image->color.multiplane.size() != 1) {
            FILE_LOG(logERROR) << "get_raw_from_image: multi-plane color image not supported";
            throw std::runtime_error("get_raw_from_image: cannot return multi-plane image");
        }
        return image->color.multiplane.front();
    }
    if (image->color.multiplane.size() != 1)
        return image->gray.plane;
    return std::get<gray_option>(opts) && !std::get<color_option>(opts) ?
        image->gray.plane : image->color.multiplane.front();
}
