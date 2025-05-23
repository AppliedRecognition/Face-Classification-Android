#pragma once

#include "input_extractor.hpp"
#include "chip_details.hpp"
#include "raw_image.hpp"
#include <raw_image/transform.hpp>
#include <dlib/image_transforms/draw.h>

namespace dlibx {
    template <typename pixel_type>
    constexpr pixel_type black_pixel() { return 0; }
    template <>
    inline dlib::rgb_pixel black_pixel<dlib::rgb_pixel>() {
        return {0,0,0};
    }

    template <typename pixel_type>
    constexpr pixel_type gray_pixel() { return 128; }
    template <>
    inline dlib::rgb_pixel gray_pixel<dlib::rgb_pixel>() {
        return {128,128,128};
    }

    /** \brief Input extractor for spoofed license classifier neural net.
     *
     * Note that this extractor expects the driver's license to be upright
     * and level.  There is no check of this condition so the classifier
     * will simply not work well if there is a non-negligable roll.
     */
    template <typename pixel_type>
    struct license_extractor final : input_extractor {
        const double radius;
        const bool normalize;

        license_extractor(std::string name,
                          unsigned width, unsigned height,
                          int radius, bool normalize)
            : input_extractor(
                move(name), width, height, raw_image::to_layout<pixel_type>()),
              radius(0 <= radius ? radius : width*(2.0/7.0)),
              normalize(normalize) {
        }

        raw_image::scaled_chip
        chip_from_pts(const std::vector<raw_image::point2f>& pts) const override {

            raw_image::point2f eye_left, eye_right;
            switch (pts.size()) {
            case 2:
                eye_left = pts.front();
                eye_right = pts.back();
                break;
            case 5:
                eye_left  = 0.5f * (pts[2] + pts[3]);
                eye_right = 0.5f * (pts[0] + pts[1]);
                break;
            case 68:
                eye_left  = 0.5f * (pts[36] + pts[39]);
                eye_right = 0.5f * (pts[42] + pts[45]);
                break;
            default:
                throw std::invalid_argument("incorrect number of landmarks");
            }
            if (eye_left.x >= eye_right.x)
                throw std::invalid_argument("license image appears to be upsidedown");
            /*
            else if (std::abs(eye_right.y() - eye_left.y()) / (eye_right.x() - eye_left.x()) > 0.125)
                FILE_LOG(logWARNING) << "license image not level";
            */
            const auto ed =
                float(std::sqrt(length_squared(eye_right - eye_left)));
            const auto mid =
                0.5f * (eye_left + eye_right) + raw_image::point2f{0,ed/2};

            const auto cx = mid.x - 0.5f;
            const auto cy = mid.y - 0.5f;
            const auto dx = (6*ed - 1) / 2;
            const auto dy = (7*ed - 1) / 2;

            dlib::chip_details cd;
            cd.rect = { cx-dx, cy-dy, cx+dx, cy+dy };
            cd.angle = 0;
            cd.rows = height;
            cd.cols = width;
            return cd;
        }

        static void normalize_brightness(const raw_image::plane& img) {
            const auto bpp = bytes_per_pixel(img);
            uint64_t sum = 0;
            auto line = img.data;
            for (auto j = img.height; j > 0; --j, line += img.bytes_per_line) {
                auto px = line;
                for (auto i = img.width; i > 0; --i, px += bpp)
                    sum += unsigned(*px);
            }
            sum /= uint64_t(img.width) * img.height;
            const auto ofs = 128 - int(sum);
            line = img.data;
            for (auto j = img.height; j > 0; --j, line += img.bytes_per_line) {
                auto px = line;
                for (auto i = img.width; i > 0; --i, px += bpp) {
                    const auto z = ofs + *px;
                    *px = uint8_t(z < 0 ? 0 : z < 256 ? z : 255);
                }
            }
        }

        /// first step of sample extraction
        raw_image::plane_ptr
        extract_chip(const raw_image::multi_plane_arg& image,
                     const raw_image::scaled_chip& cd) const {
            const auto layout =
                normalize && this->layout != raw_image::pixel::gray8 ?
                raw_image::pixel::yuv : this->layout;
            return extract_image_chip(image, cd, layout);
        }

        /// remainder of sample extraction: normalize and mask
        void finish_extract(raw_image::plane_ptr& sample) const {
            if (normalize) {
                normalize_brightness(*sample);
                if (auto p = convert(*sample, layout))
                    sample = move(p);
            }
            if (1 <= radius) {
                raw_image::fixed_dlib_image<pixel_type> dimg(*sample);
                dlib::draw_solid_circle(
                    dimg, {width/2.0,height/2.0}, radius,
                    normalize ? gray_pixel<pixel_type>() :
                    black_pixel<pixel_type>());
            }
        }

        raw_image::plane_ptr
        extract_from_chip(const raw_image::multi_plane_arg& image,
                          const raw_image::scaled_chip& cd) const override {
            auto sample = extract_chip(image,cd);
            finish_extract(sample);
            return sample;
        }
    };

    /** \brief Decode extractor description string.
     *
     * Format is "licenseWWWxHHHrRRpixel" where
     *   WWW is width,
     *   HHH is height,
     *   RR is radius of masking circle, and
     *   pixel is one of "rgb", "rgbn" or "gray".
     *
     * If "rgb" or "gray", the masking circle is black.
     * If "rgbn" image intensity (brightness) is normalized and
     * the masking circle is medium gray.
     * Note that this medium gray will become zero in the input tensor.
     *
     * If the "rRR" radius is not specified the default width*2/7.
     * For no masking circle, one must explicitly include "r0".
     */
    std::tuple<unsigned, unsigned, int, raw_image::pixel_layout, bool>
    license_decode(const std::string_view& name);

    std::unique_ptr<const input_extractor>
    license_factory(const std::string_view& name);
}
