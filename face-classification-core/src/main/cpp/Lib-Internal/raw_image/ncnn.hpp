#pragma once

#include "core.hpp"

#include <mat.h> // from ncnn

namespace raw_image {
    /** \brief Convert a grayscale or RGB image to ncnn tensor.
     */
    inline ncnn::Mat to_ncnn_rgb(single_plane_arg image) {
        if (empty(image)) return {};
        int type;
        if (bytes_per_pixel(image) <= 1)
            type = ncnn::Mat::PIXEL_GRAY2RGB;
        else {
            switch (image->layout) {
            case pixel::rgb24: type = ncnn::Mat::PIXEL_RGB; break;
            case pixel::rgba32: type = ncnn::Mat::PIXEL_RGBA2RGB; break;
            case pixel::bgr24: type = ncnn::Mat::PIXEL_BGR2RGB; break;
            case pixel::bgra32: type = ncnn::Mat::PIXEL_BGRA2RGB; break;
            default:
                throw std::invalid_argument("ncnn::Mat does not support raw_image pixel layout");
            }
        }
        return ncnn::Mat::from_pixels(
            image->data, type, int(image->width), int(image->height),
            int(image->bytes_per_line));
    }
}
