#pragma once

#include "point2.hpp"
#include <string_view>
#include <vector>

namespace raw_image {

    /** \brief Type of landmark detection.
     *
     * Note: stasm77 landmark detection has been removed, but
     * such landmarks may exist in saved (serialized) faces.
     */
    enum class detection_type {
        unknown = 0,
        v3_dlib,      ///< v3 dlib detection (hog)
        v4_dlib,      ///< v4 dlib detection (cnn)
        v5_fapi,      ///< v5 faceapi tiny face detector
        v6_rfb320,    ///< v6 rfb320 detector
        v7_retina,    ///< v7 RetinaFace detector with 5 landmarks
        v8_blaze,     ///< v8 BlazeFace detector with 6 landmarks
        haar_eyes,    ///< haar cascade eye detection
        stasm77,      ///< stasm 77 landmarks
        dlib5,        ///< dlib 5 landmarks
        dlib68,       ///< dlib 68 landmarks
        mesh68,       ///< MediaPipe FaceMesh reduced to 68 landmarks
        mesh478       ///< MediaPipe FaceMesh full 478 landmarks
    };
    std::string_view to_string(detection_type value);
    detection_type dt_from_string(std::string_view str);


    /** \brief Left and right eyes.
     *
     * Defined as the point midway between the corners of each eye.
     *
     * Viewer perspective (not subject).
     */
    struct eye_coordinates {
        point2f eye_left, eye_right;
        float eye_distance() const;  ///< distance between eyes in pixels
    };

    /** \brief Set of landmarks originating from a single detection.
     */
    struct landmark_coordinates : eye_coordinates {
        /** \brief Type of detection.
         */
        detection_type type;

        /** \brief Confidence.
         */
        float confidence = 0;

        /** \brief Landmarks.
         *
         * Face detectors v3 to v6 (inclusive) provide two landmarks
         * which are the top-left and bottom-right corners of the detected
         * bounding box.  Eye coordinates are estimated from these.
         *
         * Face detector v7 (RetinaFace) provides 7 landmarks:
         * eyes, tip of nose, mouth corners and the two bounding box corners.
         *
         * Face detector v8 (BlazeFace) provides 8 landmarks:
         * eyes, tip of nose, mouth center, tragions and
         * the two bounding box corners.
         *
         * haar_eyes has 2 landmarks which are the 2 eyes.
         *
         * dlib5, dlib68 and mesh478 provide 5, 68 and 478 landmarks
         * (respectively).
         * mesh68 is the same detector as mesh478 but reduced to
         * the 68 landmark subset.
         */
        std::vector<point2f> landmarks;

        /** \brief Constructor.
         */
        landmark_coordinates(detection_type type = detection_type::unknown) noexcept : type(type), confidence(0) {}

        /** \brief Compute eye coordinates.
         *
         * Throws exception in case of failure.
         */
        void set_eye_coordinates_from_landmarks();
    };
}
