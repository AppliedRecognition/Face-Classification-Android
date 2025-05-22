#pragma once

#include "face_types.hpp"
#include <stdext/span.hpp>

namespace raw_image {

    /** \brief Map of landmark indices to their mirrored pair.
     *
     * For each landmark i, if j = span[i], then
     * either i == j is center landmark
     * or i and j are the same landmark on opposite sides of the face.
     */
    stdx::span<const unsigned> mirrored_pairs(detection_type dt);

    /** \brief Get indices for triangulation of landmarks.
     */
    std::vector<std::array<unsigned short, 3> > triangles(detection_type dt);

    /** \brief Map of landmark indices selected from larger set.
     *
     * For each landmark i in "to", landmark j = span[i]
     * is the corresponding landmark in "from".
     */
    stdx::span<const unsigned>
    landmark_subset(detection_type from, detection_type to);

    /** \brief Interpolate landmarks from larger set.
     *
     * This method provides a greater variety of conversions because
     * it will also interpolate landmarks in addition to copying them.
     */
    void landmark_subset(const landmark_coordinates& from,
                         detection_type to, landmark_coordinates& dest);

    /** \brief Extract or interpolate center of eyes from landmarks.
     */
    eye_coordinates eyes_subset(const landmark_coordinates& from);
}
