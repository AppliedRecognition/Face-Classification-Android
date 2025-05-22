#pragma once

#include "types.hpp"

#include <open3d/geometry/Image.h>

#include <stdexcept>

namespace open3d {
    namespace geometry {
        /** \brief Create raw_image::plane from open3d Image object.
         *
         * The returned raw_image::plane shares the same pixels as the Image.
         * Lifetime of the pixel data is managed by the Image object.
         */
        inline auto to_raw_image(const Image& img) {
            raw_image::plane p;
            if (img.HasData()) {
                p.width = img.width_;
                p.height = img.height_;
                p.bytes_per_line = img.BytesPerLine();
                p.data = img.PointerAs<uint8_t>();
                p.layout =
                    [&](){
                        switch (img.bytes_per_channel_) {
                        case 1:
                            switch (img.num_of_channels_) {
                            case 1: return raw_image::pixel::gray8;
                            case 3: return raw_image::pixel::rgb24;
                            case 4: return raw_image::pixel::rgba32;
                            }
                            break;
                        case 2:
                            switch (img.num_of_channels_) {
                            case 1: return raw_image::pixel::a16_le;
                            }
                            break;
                        }
                        throw std::invalid_argument(
                            "unsupported open3d image type");
                    }();
            }
            return p;
        }
    }
}

namespace raw_image {
    /** \brief Create raw_image::plane from open3d Image object.
     *
     * Like the open3d::geometry::to_raw_image() method above, but
     * this version ensures the layout is as specified.
     * An invalid_argument exception will be thrown if the layout
     * specified does not have the correct number of bytes per pixel.
     */
    inline plane to_raw_image(const open3d::geometry::Image& img,
                              pixel_layout layout) {
        raw_image::plane p;
        p.layout = layout;
        if (img.HasData()) {
            const auto bpp = bytes_per_pixel(layout);
            const auto ubpc = unsigned(img.bytes_per_channel_);
            const auto unc = unsigned(img.num_of_channels_);
            if (img.bytes_per_channel_ <= 0 || bpp < ubpc ||
                img.num_of_channels_ <= 0   || bpp < unc  ||
                ubpc * unc != bpp)
                throw std::invalid_argument("unsupported open3d image type or incorrect number of channels");
            p.width = img.width_;
            p.height = img.height_;
            p.bytes_per_line = img.BytesPerLine();
            p.data = img.PointerAs<uint8_t>();
        }
        return p;
    }
}
