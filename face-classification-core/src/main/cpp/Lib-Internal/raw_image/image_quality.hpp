#pragma once

#include "reader.hpp"

namespace raw_image {

    /** \brief Brightness, contrast, sharpness and gradients.
     *
     * Brightness is the mean pixel value.
     * Contrast is standard deviation of pixel values.
     * Sharpness is standard deviation of Laplacian divided by contrast.
     * The sharpness is scaled by 100 so its value has a range similar to
     * the contrast.
     * Horizontal gradient is the difference in brightness between the
     * left and right halves of the image.
     * Vertical gradient is the difference in brightness between the
     * top and bottom halves of the image.
     */
    struct bcsg_result {
        float brightness, contrast, sharpness, horz, vert;
    };

    /** \brief Compute brightness, contrast, sharpness and gradients.
     */
    bcsg_result bcsg(reader& src);
}
