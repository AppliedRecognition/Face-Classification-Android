#pragma once

#include "input_extractor.hpp"
#include "raw_image.hpp"
#include <raw_image/adjust.hpp>

namespace dlibx {
    /** \brief Input extractor to extract a rectangle rotated as needed.
     *
     * This input extractor requires 2 points as input: top-left corner
     * and bottom-right corner.  
     * It is an error for these 2 points to either have the same x value
     * or the same y value.
     *
     * The two points define a rectangle within the image which is then
     * extracted and resized to fit the output dimensions.  
     * The rectangle is also rotate by a multiple of 90 degrees so that
     * the first point is the top-left of the output and the second is
     * the bottom-right.
     */
    class box_extractor : public input_extractor {
    public:
        const bool normalize;
        box_extractor(std::string name, unsigned width, unsigned height,
                      raw_image::pixel_layout layout, bool normalize)
            : input_extractor(move(name), width, height, layout),
              normalize(normalize) {
        }

        raw_image::scaled_chip
        chip_from_pts(const std::vector<raw_image::point2f>& pts) const override {
            if (pts.size() != 2)
                throw std::invalid_argument("incorrect number of landmarks");
            dlib::chip_details cd;
            cd.rows = height;
            cd.cols = width;
            using fpoint = dlib::vector<float,2>;
            const fpoint tl = raw_image::round_from(pts.front());
            const fpoint br = raw_image::round_from(pts.back());
            auto dx = std::abs(tl.x() - br.x()) / 2;
            auto dy = std::abs(tl.y() - br.y()) / 2;
            if (dx < 1 || dy < 1)
                throw std::invalid_argument("landmarks must define a non-empty rectangle");
            if (tl.x() > br.x()) {
                if (tl.y() > br.y())
                    cd.angle = M_PI;
                else {
                    cd.angle = M_PI/2;
                    std::swap(dx,dy);
                }
            }
            else if (tl.y() > br.y()) {
                cd.angle = -M_PI/2;
                std::swap(dx,dy);
            }
            const auto cx = (tl.x() + br.x()) / 2;
            const auto cy = (tl.y() + br.y()) / 2;
            cd.rect = { cx - dx, cy - dy, cx + dx, cy + dy };
            return cd;
        }

        static void normalize_bc(const raw_image::plane& img) {
            const auto bc = measure_brightness_contrast(img);
            static constexpr auto target_contrast = 48;
            const auto alpha = target_contrast / std::max(1.0f, bc.contrast);
            const auto beta = 128 - bc.brightness * alpha;
            linear_adjust(raw_image::reader::construct(img),alpha,beta)->copy_to(img);
        }

        /// first step of sample extraction
        raw_image::plane_ptr
        extract_chip(const raw_image::multi_plane_arg& image,
                     const raw_image::scaled_chip& cd) const {
            const auto layout =
                normalize && bytes_per_pixel(this->layout) > 1 ?
                raw_image::pixel::yuv : this->layout;
            return extract_image_chip(image, cd, layout);
        }

        /// remainder of sample extraction: normalize and mask
        void finish_extract(raw_image::plane_ptr& sample) const {
            if (normalize) {
                normalize_bc(*sample);
                if (auto p = convert(*sample, layout))
                    sample = move(p);
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
     * Format is "boxWWWxHHHpixel" where
     *   WWW is width,
     *   HHH is height, and
     *   pixel is one of "rgb", "yuv" or "gray"
     *   (pixel may have an 'n' suffix to apply normalization).
     */
    std::tuple<unsigned, unsigned, raw_image::pixel_layout, bool>
    box_decode(const std::string_view& name);
        
    std::unique_ptr<const input_extractor>
    box_factory(const std::string_view& name);
}
