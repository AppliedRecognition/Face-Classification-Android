#pragma once

#include "types.hpp"
#include "image_size.hpp"
#include <stdext/arg.hpp>
#include <stdext/span.hpp>
#include <stdext/options_tuple.hpp>
#include <string>
#include <string_view>
#include <ostream>

namespace raw_image {
    /** \brief Single plane argument to method.
     *
     * This type is generally only used as the argument for a method.
     * It accepts either the reference const plane& or
     * a pointer plane const* (including smart pointers like std::unique_ptr).
     *
     * Note that objects of this type do not contain the plane or pixel data,
     * only a reference (pointer) to them.
     */
    using single_plane_arg = stdx::arg<const plane>;

    /** \brief Multi plane argument to method.
     *
     * This type is generally only used as the argument for a method.
     * It accepts a variety of reference to plane, smart pointer to plane,
     * and contiguous containers of planes (ie. array or vector).
     *
     * Note that objects of this type do not contain the plane or pixel data,
     * only a reference (pointer) to them.
     */
    using multi_plane_arg = stdx::spanarg<const plane>;

    // forward declare
    class reader;

    /** \brief Rotate option.
     *
     * If rotate > 0, rotate image by multiple of 90 degrees.
     * If rotate & 4, mirror image before rotation.
     */
    enum class rotate : unsigned {};

    /** \brief Bytes needed for plane struct (rounded up to multiple of 16).
     */
    constexpr unsigned plane_struct_padded_size = 1 + ((sizeof(plane)-1)|15);

    /** \brief Test if smart pointer object manages (contains) pixel buffer.
     *
     * The plane_ptr returned from create() and all other raw_image methods
     * that return a newly allocated image will have a managed pixel buffer.
     * Thus, this method will return true for those objects.
     */
    template <typename U>
    inline std::enable_if_t<std::is_convertible_v<decltype(std::declval<U>().get()), const plane*>, bool>
    manages_pixel_buffer(const U& smart_pointer) {
        const plane* img = smart_pointer.get();
        return img->data == reinterpret_cast<const unsigned char*>(img) + plane_struct_padded_size;
    }

    /** \brief String description of pixel layout.
     *
     * Note that the returned string is all caps (e.g. "RGB24").
     */
    std::string to_string(pixel_layout);
    inline std::ostream& operator<<(std::ostream& out, pixel_layout pl) {
        return out << to_string(pl);
    }

    /** \brief String description of color class.
     */
    std::string to_string(color_class);
    inline std::ostream& operator<<(std::ostream& out, color_class cc) {
        return out << to_string(cc);
    }

    /** \brief Number of bytes needed per pixel.
     */
    inline unsigned bytes_per_pixel(single_plane_arg image) {
        return image ? bytes_per_pixel(image->layout) : 0;
    }

    /** \brief Test if image is empty.
     *
     * The image is empty if it has no planes, or if either width or height
     * is zero.
     * In the case of a multi-plane image, only the first plane is assessed.
     */
    inline bool empty(const multi_plane_arg& mp) {
        return mp.empty() || mp.front().width <= 0 || mp.front().height <= 0;
    }

    /** \brief Dimensions of image after rotate and scale.
     *
     * This method returns the dimensions of the image after the rotate
     * and scale specified within the image struct have been applied.
     *
     * In the case of a multi-plane image, the size is of the first plane.
     * If the image is empty, then 0x0 is returned.
     */
    image_size dimensions(const multi_plane_arg& image);

    /** \brief Assess image structure for validity and return description
     * of error.
     *
     * An empty image (no planes or width == height == 0) is considered valid,
     * and the other fields are not checked in this case.
     * The case where width == 0 or height == 0, but not both, is not valid.
     *
     * \returns first error found or empty string if image is valid
     */
    std::string_view describe_error(const multi_plane_arg& image);

    /** \brief Verify validity of image structure.
     *
     * This method behaves the same as calling describe_error() and
     * throwing an exception if the result is non-empty.
     *
     * The method string will be included in the log message.
     *
     * \throws invalid_argument exception if image is not valid.
     */
    void throw_if_invalid(
        const multi_plane_arg& image, std::string_view method = {});

    /** \brief Verify validity of non-empty image.
     *
     * The method string will be included in the log message.
     *
     * \throws invalid_argument if either empty or invalid.
     */
    void throw_if_invalid_or_empty(
        const multi_plane_arg& image, std::string_view method = {});

    /** \brief Return description of image for diagnostic purposes.
     */
    std::string diag(single_plane_arg image);
    inline std::ostream& operator<<(std::ostream& out, single_plane_arg image) {
        return out << diag(image);
    }
    
    /** \brief Create uninitialized image.
     *
     * The memory needed for the pixel data is also contained and managed by
     * the returned pointer object.
     */
    plane_ptr create(unsigned width, unsigned height, pixel_layout layout);

    /** \brief Copy image and optionally change pixel layout or rotate.
     *
     * The width, height, rotate and scale of the output image will
     * match the first of the input planes.
     * If the rotate option is provided, then width, height and rotate
     * will be modified as dictated by the rotation done.
     *
     * Supported multi-plane images include: <ul>
     *   <li>2-plane Y8 + UV16 and</li>
     *   <li>3-plane Y8 + U8 + V8.</li>
     * </ul>
     * The U and V planes may either have the same dimensions as the Y plane,
     * or be at half width/height.
     */
    plane_ptr
    copy_with_opts(const multi_plane_arg& from,
                   const stdx::options_tuple<rotate,pixel_layout>& opts);
    template <typename... Opts>
    inline plane_ptr
    copy(const multi_plane_arg& from, Opts&&... opts) {
        return copy_with_opts(from, { std::forward<Opts>(opts)... } );
    }

    /** \brief Copy from reader with optional rotate.
     */
    plane_ptr copy(stdx::arg<reader> from, rotate rot = rotate{0});

    /** \brief Copy pixels and optionally change pixel layout or rotate.
     *
     * "To" image must have same dimensions as "from" image (after rotate).
     * Pixels must have seperate memory allocations.
     *
     * The rotate value is in multiples of 90 degrees.
     * If rotate & 4, then mirror image before rotation.
     * Note that this method does not modify the value of rotate
     * in the destination image.
     */
    void copy_pixels(
        const multi_plane_arg& from, single_plane_arg to, unsigned rotate = 0);

    /** \brief Change the pixel layout of the image in-place.
     *
     * If the pixel layout cannot be changed in place (insufficient memory),
     * then a new image is allocated and returned.
     * In this case, the original image is not modified.
     *
     * If the image is modified in place then only the layout field
     * is modified (ie. bytes_per_line remains as is).
     *
     * \return new image or null if image was modified in place
     */
    plane_ptr convert(plane& image, pixel_layout new_layout);

    /** \brief Shallow crop of image.
     */
    plane crop(single_plane_arg image,
               unsigned x, unsigned y, unsigned w, unsigned h);


    /** \brief Channel names for copy_channel().
     */
    enum class channel {
        ch0 = 0, ch1 = 1, ch2 = 2, ch3 = 3,
        y = -1,
        u = -2,
        v = -3,
        r = -4, red = -4,
        g = -5, green = -5,
        b = -6, blue = -6,
        alpha = -7
    };

    /** \brief Copy single channel.
     *
     * "To" image must have same dimensions as "from" image.
     * "To" and "from" image may be the same image.
     * Other channels in "to" image are not modified.
     */
    void copy_channel(single_plane_arg from, channel from_ch,
                      single_plane_arg to, channel to_ch = channel(0));

    /** \brief Method to compute grayscale pixel intensity from pixel layout; 
     */
    auto gray8_from_pixel(pixel_layout cs)
        -> unsigned char(*)(const unsigned char*);
}
