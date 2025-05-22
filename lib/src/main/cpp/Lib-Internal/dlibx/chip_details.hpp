#pragma once

#include <dlib/image_transforms/interpolation.h>

namespace dlibx {

    /** \brief Compute chip_details object from landmarks.
     *
     * Extends the dlib version of the same method by adding support for
     * two eyes only and the RetinaFace set of 7 landmarks.
     * This method also supports dlib5 and dlib68 landmarks by forwarding
     * to the dlib version.
     */
    dlib::chip_details
    get_face_chip_details(const std::vector<dlib::dpoint>& pts,
                          unsigned long size, double padding);

    /** \brief Compute chip_details object from eye coordinates.
     *
     * Left and right are relative to the viewer (not the subject).
     */
    dlib::chip_details
    get_face_chip_details(dlib::dpoint eye_left, dlib::dpoint eye_right,
                          unsigned long size, double padding);
}
