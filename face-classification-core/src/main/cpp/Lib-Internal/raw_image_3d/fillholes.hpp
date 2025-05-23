#pragma once

#include <raw_image/types.hpp>
#include <stdext/options_tuple.hpp>

namespace raw_image {
    /** \brief Extrapolate option.
     */
    struct extrapolate_tag;
    using extrapolate_option = stdx::option_bool<extrapolate_tag>;
    const extrapolate_option interpolate{false};
    const extrapolate_option extrapolate{true};


    /** \brief Scan image looking for pixels with tofill value (a hole) and
     * interpolate neighbouring pixels to fill the hole.
     *
     * If extrapolate option is specified, then holes starting from an edge
     * of the image will be filled with a constant pixel value determined
     * from the first non-hole pixel found.
     *
     * If interpolate is selected, then holes are only filled if there are
     * non-hole pixels on either side (either up and down or left and right).
     *
     * Implementations are provided for uint8_t and uint16_t.
     */
    template <typename CHTYPE>
    void in_place_fill_holes(const plane& img, CHTYPE const* tofill,
                             extrapolate_option = extrapolate);


    /** \brief Fill holes using bytewise interpolation.
     *
     * The tofill value defines what a hole is.
     * Only the first BPP bytes are used.
     *
     * This method may not fill all the way to the edge of the image.
     * If this is needed, place a 1 pixel wide border around the image
     * with a non-hole pixel value to interpolate to.
     */
    void in_place_fill_bytes(
        const plane& img, std::array<uint8_t,4> tofill);
}
