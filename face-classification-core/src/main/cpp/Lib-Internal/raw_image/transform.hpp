#pragma once

#include "core.hpp"
#include <array>

namespace raw_image {

    /** \brief Flip image top to bottom.
     *
     * This method will modify / output the rotate value within
     * the raw image object so that it maintains its meaning.
     * That is, it's the rotation required to make the image upright.
     *
     * Flip is the same as rotate by 6.
     */
    void in_place_flip(plane& img);

    template <typename... Opts>
    inline plane_ptr
    copy_flip(const multi_plane_arg& img, Opts&&... opts) {
        return copy(img, std::forward<Opts>(opts)..., rotate(6));
    }

    /** \brief Mirror image left to right.
     *
     * This method will modify / output the rotate value within
     * the raw image object so that it maintains its meaning.
     * That is, it's the rotation required to make the image upright.
     *
     * Mirror is the same as rotate by 4.
     */
    void in_place_mirror(plane& img);

    template <typename... Opts>
    inline plane_ptr
    copy_mirror(const multi_plane_arg& img, Opts&&... opts) {
        return copy(img, std::forward<Opts>(opts)..., rotate(4));
    }

    /** \brief Transpose image top-right to bottom-left.
     *
     * This method will modify / output the rotate value within
     * the raw image object so that it maintains its meaning.
     * That is, it's the rotation required to make the image upright.
     *
     * Transpose is the same as rotate by 5.
     */
    void in_place_transpose(plane& img);

    template <typename... Opts>
    inline plane_ptr
    copy_transpose(const multi_plane_arg& img, Opts&&... opts) {
        return copy(img, std::forward<Opts>(opts)..., rotate(5));
    }

    /** \brief Rotate image by multiple of 90 degrees.
     *
     * If rotate & 4, then mirror image before rotation.
     *
     * This method will also modify / output the rotate value within
     * the raw image object so that it maintains its meaning.
     * That is, it's the rotation required to make the image upright.
     */
    void in_place_rotate(plane& img, unsigned rotate);
    inline void in_place_rotate(plane& img) {
        in_place_rotate(img, img.rotate);
    }

    template <typename... Opts>
    inline plane_ptr
    copy_rotate(const multi_plane_arg& img,
                unsigned rot, Opts&&... opts) {
        return copy(img, std::forward<Opts>(opts)..., rotate(rot));
    }
    inline plane_ptr copy_rotate(const plane& img) {
        return copy(img, rotate(img.rotate));
    }

    /** \brief Interpolation options.
     */
    enum class interpolation_type {
        nearest, area, bilinear
    };
    using inter = interpolation_type;

    /** \brief Resize image to specified dimensions.
     *
     * The rotate and scale values present in the input image
     * will be set the same in the output image.
     */
    plane_ptr copy_resize(const multi_plane_arg& img,
                          unsigned width, unsigned height,
                          pixel_layout layout,
                          interpolation_type i = inter::bilinear);
    inline plane_ptr copy_resize(single_plane_arg img,
                                 unsigned width, unsigned height,
                                 interpolation_type i = inter::bilinear) {
        return copy_resize(img, width, height, img->layout, i);
    }

    /** \brief Extract region.
     *
     * Destination image will have dimensions (dest_width,dest_height).
     *
     * In source image, region is defined by center (cx,cy),
     * dimensions (w,h) and rotation angle in degrees.
     *
     * This method takes the value of image.scale into account in
     * that it will divide cx, cy, w, and h by 2^image.scale before
     * using them to extract the region.
     *
     * This method also takes into account the value of image.rotate.
     * The source image is considered to be image after the specified
     * mirror and rotate are applied.
     * Therefore, (cx,cy) is relative to image after mirror and rotate.
     *
     * To compute (cx,cy) from region (lx,ty,w,h), where lx is left x
     * and ty is top y, the result is cx = lx + w/2 and cy = ty + h/2.
     *
     * As an example, to extract a deep copy of the entire image as it
     * would appear after image.scale and image.rotate, do
     *   extract_region(image, w/2.0f, h/2.0f, w, h, 0, w, h);
     * Where (w,h) are the dimensions after scale and rotate.
     */
    plane_ptr
    extract_region(const multi_plane_arg& image,
                   float cx, float cy, float w, float h, float angle,
                   unsigned dest_width, unsigned dest_height,
                   pixel_layout dest_layout);
    inline plane_ptr
    extract_region(const plane& image,
                   float cx, float cy, float w, float h, float angle,
                   unsigned dest_width, unsigned dest_height) {
        return extract_region(image, cx, cy, w, h, angle,
                              dest_width, dest_height, image.layout);
    }



    /** \brief Create multi-plane android compatible NV21 Y-VU image.
     *
     * This method is intended for testing uses.
     *
     * The input image is cropped such that both width and height are
     * multiples of 8.
     *
     * The pixels are stored in a single packed buffer with the Y plane
     * first followed by the VU plane.
     *
     * This method allocates sufficient memory for both the array and
     * the pixel buffer (managed by the returned smart pointer).
     */
    std::unique_ptr<std::array<plane,2> >
    create_nv21(plane image);


    /** \brief Matrix invertion.
     *
     * The input image must be square and have layout f32.
     */
    plane_ptr matrix_inverse(const plane& mat);
}
