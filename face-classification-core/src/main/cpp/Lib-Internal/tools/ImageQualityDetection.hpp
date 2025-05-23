#pragma once

//
//  ImageQualityDetection.hpp
//  VerIDCore
//
//  Created by Jakub Dolejs on 01/02/2019.
//  Copyright © 2019 Applied Recognition. All rights reserved.
//

namespace verid {
    /** \brief Compute variance of Laplacian.
     *
     * Note that this method does not correct for contrast, so
     * greater contrast gives greater sharpness.
     */
    double sharpnessOfImage(const void* grayscaleBuffer,
                            unsigned width, unsigned height);

    
    /** \brief Compute brightness, contrast and sharpness.
     *
     * Brightness is the mean pixel value.
     * Contrast is standard deviation of pixel values.
     * Sharpness is standard deviation of Laplacian divided by contrast.
     * The sharpness is scaled by 100 so its value has a range similar to
     * the contrast.
     */
    struct bcs_result {
        double brightness, contrast, sharpness, horz, vert;
    };
    bcs_result brightnessContrastAndSharpnessOfImage(
        const void* grayscaleBuffer, unsigned width, unsigned height);
}
