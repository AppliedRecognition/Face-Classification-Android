#pragma once

#include "coordinates.hpp"
#include "detection_settings.hpp"

#include <raw_image/image_size.hpp>
#include <raw_image/face_types.hpp>

#include <memory>
#include <string>
#include <utility>  // for std::pair
#include <vector>


namespace stdx {
    class binary;
}
namespace json {
    class value;
}
namespace core {
    class context;
}
namespace raw_image {
    struct plane;
}

namespace det {
    using detection_type = raw_image::detection_type;
    using dt = detection_type;

    using eye_coordinates = raw_image::eye_coordinates;

    /** \brief Coordinates and confidence from a single detection.
     */
    struct detected_coordinates : raw_image::landmark_coordinates {

        using landmark_coordinates::landmark_coordinates;

        /** \brief Deserialize from json.
         *
         * Supported values are:<ul>
         *   <li>array of 5, 7 or 68 landmarks</li>
         *   <li>object {
         *           "t":  type,
         *           "el": eye_left,
         *           "er": eye_right,
         *           "c":  confidence,
         *           "lm": landmarks_array
         *       }</li>
         * </ul>
         */
        explicit detected_coordinates(const json::value& val);
    };

    /** \brief All detections associated with a single face.
     *
     * The individual detections are in the order they were detected and
     * are generally from coarsest to finest detection.
     * Therefore, back() is assumed to contain the most precise set of
     * coordinates.
     */
    struct face_coordinates : std::vector<detected_coordinates> {
        using std::vector<detected_coordinates>::vector;

        /** \brief Deserialize.
         *
         * Value may be one of the output of to_json(), to_binary(), or
         * to_binary() as a base64 encoded string.
         */
        explicit face_coordinates(const json::value& val);

        /** \brief Construct from single detected_coordinates element.
         */
        explicit face_coordinates(detected_coordinates dc)
            : std::vector<detected_coordinates>{std::move(dc)} {}

        /// implicit conversion to back()
        operator detected_coordinates&() &;
        operator detected_coordinates&&() &&;
        operator const detected_coordinates&() const&;
        operator const detected_coordinates&&() const&&;
    };

    /** \brief Serialize face_coordinates object.
     *
     * Formats: <ul>
     *   <li>0: AMF3 compressed</li>
     *   <li>1: AMF3 raw</li>
     *   <li>2: JSON compressed</li>
     *   <li>3: JSON raw</li>
     * </ul>
     *
     */
    json::value to_json(const face_coordinates& fc);
    stdx::binary to_binary(const face_coordinates& fc, unsigned format = 0);

    /** \brief List of faces.
     */
    using face_list_type = std::vector<face_coordinates>;

    /** \brief Pitch, yaw and roll.
     */
    struct face_pose_type {
        float pitch, yaw, roll;   // degrees
    };

    /** \brief Internal image object.
     */
    struct image_struct;
    struct image_deleter { void operator()(const image_struct*); };
    using image_type = std::unique_ptr<const image_struct, image_deleter>;
}
