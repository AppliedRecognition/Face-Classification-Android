#pragma once

#include "reader.hpp"
#include <stdext/rounding.hpp>

namespace raw_image {

    /** \brief Linear adjustment of pixel intensity line-by-line.
     *
     * Each pixel will have it's intensity x adjusted to y = x*alpha + beta.
     * For a single channel, the adjustment is as stated.
     * For a multi-channel image, each pixel is converted to YUV, then the
     * Y channel is adjusted as stated, and finally the pixel is converted
     * back to the source format (or the specified output_layout).
     */
    std::unique_ptr<reader>
    linear_adjust(std::unique_ptr<reader> src, float alpha, float beta);
    std::unique_ptr<reader>
    linear_adjust(std::unique_ptr<reader> src, float alpha, float beta,
                  pixel_layout output_layout);

    /** \brief Linear adjustment of image in place.
     *
     * If the image is 32-bits per pixel including an alpha channel, the
     * alpha channel is not modified.
     */
    void in_place_linear_adjust(const plane& image, float alpha, float beta);


    /** \brief Blending of two images line-by-line.
     *
     * If x1 is pixel value from src1 and x2 is pixel from src2, then
     * output pixel value is y = x1*alpha1 + x2*alpha2 + beta.
     * For multi-channel images, each channel is blended independantly.
     *
     * The input images must have the pixel layout.
     * If their dimensions don't match then the output will have width equal
     * to the minimum of widths and the same for height.
     */
    std::unique_ptr<reader>
    blend(std::unique_ptr<reader> src1, float alpha1,
          std::unique_ptr<reader> src2, float alpha2, float beta = 0);


    /** \brief Brightness and contrast.
     */
    struct bc_result {
        float brightness;
        float contrast;
        unsigned count;  ///< number of pixels used to compute result
    };

    /** \brief Measure brightness and contrast of image.
     *
     * For images with two or more channels, the pixels are converted to a
     * YUV format and only the Y channels is measured.
     *
     * Brightness is the mean intensity (Y channel) value and contrast
     * is the standard deviation.
     *
     * If area equal to or greater than 1, all pixels are used in the
     * calculation.  If area < 1, then only pixels within an ellipse centered
     * within the image are used.
     * Specifically, if area = M_PI/4, approximately 0.7854, then the largest
     * inscribed ellipse is used.  Of course, if width == height, then this
     * ellipse is actually a circle.
     */
    bc_result
    measure_brightness_contrast(single_plane_arg image, float area = 1.0f);

    /** \brief Measure brightness only.
     */
    float measure_brightness(single_plane_arg image);

    /** \brief Measure brightness and contrast, and then apply correction.
     *
     * \sa measure_brightness_contrast() for description of area
     *
     * \returns brightness and contrast before adjustment
     */
    inline bc_result in_place_adjust_contrast_brightness(
        const plane& image,
        float target_contrast = 48,
        float target_brightness = 128,
        float area = 1.0f) {
        auto bc = measure_brightness_contrast(image, area);
        const auto alpha = target_contrast / std::max(1.0f, bc.contrast);
        const auto beta = target_brightness - bc.brightness * alpha;
        in_place_linear_adjust(image, alpha, beta);
        return bc;
    }

    /** \brief Measure brightness and contrast, and then apply correction.
     *
     * \returns reader for adjusted image
     */
    inline std::unique_ptr<reader>
    adjust_contrast_brightness(
        single_plane_arg image,
        float target_contrast = 48,
        float target_brightness = 128) {
        const auto bc = measure_brightness_contrast(image);
        const auto alpha = target_contrast / std::max(1.0f, bc.contrast);
        const auto beta = target_brightness - bc.brightness * alpha;
        return linear_adjust(reader::construct(image), alpha, beta);
    }
    inline std::unique_ptr<reader>
    adjust_contrast_brightness(
        single_plane_arg image, pixel_layout output_layout,
        float target_contrast = 48,
        float target_brightness = 128) {
        const auto bc = measure_brightness_contrast(image);
        const auto alpha = target_contrast / std::max(1.0f, bc.contrast);
        const auto beta = target_brightness - bc.brightness * alpha;
        return linear_adjust(reader::construct(image), alpha, beta, output_layout);
    }


    /** \brief Rotate UV color plane.
     *
     * If the input reader is not already UV or YUV, then the pixels are
     * first converted to YUV.
     *
     * \param src input image
     * \param color_angle rotation angle radians of uv plane
     * \returns reader producing UV or YUV pixels.
     */
    std::unique_ptr<reader>
    rotate_yuv(std::unique_ptr<reader> src, float color_angle);


    /** \brief Helper class for adding noise to image.
     */
    template <unsigned BPP, typename GEN>
    struct add_noise_quads {
        GEN gen;
        void operator()(
            uint8_t* dest, const uint8_t* src, unsigned nquads) const {
            for ( ; nquads > 0; --nquads)
                for (auto n = 4; n > 0; --n, ++dest, ++src) {
                    *dest = stdx::round_from(*src + gen());
                    for (auto k = BPP-1; k > 0; --k)
                        *++dest = *++src;
                }
        }
    };

    /** \brief Add noise to image.
     *
     * If the input reader is not grayscale or YUV, then the pixels are
     * first converted to YUV.
     *
     * \param src input image
     * \param noise_gen function producing delta to add to Y channel
     * \returns reader producing grayscale or YUV pixels.
     */
    template <typename GEN>
    std::unique_ptr<reader>
    add_noise(std::unique_ptr<reader> src, GEN noise_gen) {
        if (!src) throw std::invalid_argument("empty image");
        const auto layout = src->layout();
        using std::move;
        switch (src->bytes_per_pixel()) {
        case 1:
            return transform_quads(
                move(src), layout, add_noise_quads<1,GEN>{move(noise_gen)});
        case 3: {
            const auto ccls = to_color_class(layout);
            if (ccls == cc::yuv_jpeg || ccls == cc::yuv_nv21)
                break;
        } [[fallthrough]];
        case 2:
        case 4:
            src = convert(move(src), raw_image::pixel::yuv);
            break;
        default:
            throw std::logic_error("unexpected bytes per pixel");
        }
        return transform_quads(
            move(src), layout, add_noise_quads<3,GEN>{move(noise_gen)});
    }


    /** \brief Matrix multiply.
     *
     * Multiply (inner product) all rows in src1 by all rows in src2,
     * as in the operation operation src1 * transpose(src2).
     * Must have src1.width == src2.width.
     * The output will have dimensions width = src2.height and
     * height = src1.height.
     * This method only works with the f32 pixel layout.
     */
    std::unique_ptr<reader>
    matrix_multiply(std::unique_ptr<reader> src1, const plane& src2);
}
