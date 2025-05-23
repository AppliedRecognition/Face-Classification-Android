#pragma once

#include "core.hpp"

#if __has_include(<opencv2/core/ocl.hpp>)
#include <opencv2/core/ocl.hpp>
#else
#include <opencv2/core/core.hpp>
namespace cv { namespace ocl { constexpr void useOpenCL() {} } }
#endif

#include <stdexcept>

namespace cv {
    /** \brief Create raw_image::plane from cv::Mat.
     *
     * The returned raw_image::plane shares the same pixels as the cv::Mat.
     * Lifetime of the pixel data is managed by the cv::Mat object.
     */
    inline raw_image::plane to_raw_image(const Mat& img) {
        if (img.rows < 0 || img.cols < 0)
            throw std::invalid_argument("cv::Mat has negative dimension");
        raw_image::plane r;
        unsigned bpp;
        switch (img.type()) {
        case CV_8UC1: r.layout = raw_image::pixel::gray8,  bpp = 1; break;
        case CV_8UC3: r.layout = raw_image::pixel::bgr24,  bpp = 3; break;
        case CV_8UC4: r.layout = raw_image::pixel::bgra32, bpp = 4; break;
        default:
            throw std::invalid_argument("unsupported cv::Mat image type");
        }
        r.data = img.data;
        r.width = unsigned(img.cols);
        r.height = unsigned(img.rows);
        r.bytes_per_line = unsigned(img.step);
        if (r.bytes_per_line < bpp * r.width)
            throw std::invalid_argument("cv::Mat has insufficient step");
        return r;
    }
}

namespace raw_image {

    /** \brief Create raw_image::plane from cv::Mat.
     *
     * Like the cv::to_raw_image() method, but this version ensures the
     * layout is as specified.
     * An invalid_argument exception will be thrown if the layout
     * specified does not have the correct number of channels.
     */
    inline plane to_raw_image(const cv::Mat& img, pixel_layout layout) {
        if (img.rows < 0 || img.cols < 0)
            throw std::invalid_argument("cv::Mat has negative dimension");
        const auto bpp = bytes_per_pixel(layout);
        if (img.type() != int(CV_8UC(bpp)))
            throw std::invalid_argument("unsupported cv::Mat image type or incorrect number of channels");
        plane r;
        r.width = unsigned(img.cols);
        r.height = unsigned(img.rows);
        r.data = img.data;
        r.bytes_per_line = unsigned(img.step);
        if (r.bytes_per_line < bpp * r.width)
            throw std::invalid_argument("cv::Mat has insufficient step");
        r.layout = layout;
        return r;
    }

    /** \brief Create cv::Mat from raw_image.
     *
     * A valid cv::Mat object is created for all layout, but only
     * GRAY8 and BGR24 are opencv kosher.
     */
    inline cv::Mat to_Mat(single_plane_arg image) {
        if (!image) throw std::invalid_argument("raw_image nullptr");
        return {
            int(image->height),
            int(image->width),
            CV_8UC(int(bytes_per_pixel(image))),
            image->data,
            image->bytes_per_line
        };
    }

    /** \brief Initialize OpenCV for multithreaded use.
     *
     * On some platforms if resize(), extract_region() or extract_image_chip()
     * are called for the first time from multiple threads simultaneously,
     * the application will crash.
     *
     * This method may be called from main() before making use of multiple
     * threads to initialize OpenCL.
     *
     * This method calls cv::ocl::useOpenCL().
     */
    inline void init_opencv() {
        cv::ocl::useOpenCL();
    }
}
