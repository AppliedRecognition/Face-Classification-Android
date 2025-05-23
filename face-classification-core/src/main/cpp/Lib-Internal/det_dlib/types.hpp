#pragma once

#include <det/types.hpp>

namespace det {

    /** \brief Classifier name, path and model.
     *
     * \see classifiers.hpp
     */
    struct classifier_model_type;

    /** \brief Result of classifier assessment.
     */
    using classifier_result_pair =
        std::pair<classifier_model_type const*, std::vector<float> >;

    /** \brief Face coordinates and assessed classifiers.
     */
    struct face_coordinates_with_classifiers : face_coordinates {
        using face_coordinates::face_coordinates;

        std::vector<classifier_result_pair> classifiers;

        face_coordinates_with_classifiers(
            face_coordinates_with_classifiers&&) = default;
        face_coordinates_with_classifiers(
            const face_coordinates_with_classifiers&) = default;

        face_coordinates_with_classifiers(face_coordinates&& other)
            : face_coordinates(std::move(other)) {}
        face_coordinates_with_classifiers(const face_coordinates& other)
            : face_coordinates(other) {}

        face_coordinates_with_classifiers&
        operator=(face_coordinates_with_classifiers&&) = default;
        face_coordinates_with_classifiers&
        operator=(const face_coordinates_with_classifiers&) = default;

        face_coordinates_with_classifiers&
        operator=(face_coordinates&& fc) {
            static_cast<face_coordinates&>(*this) = std::move(fc);
            classifiers.clear();
            return *this;
        }
        face_coordinates_with_classifiers&
        operator=(const face_coordinates& fc) {
            static_cast<face_coordinates&>(*this) = fc;
            classifiers.clear();
            return *this;
        }

        /** \brief Deserialize.
         *
         * Value may be one of the output of to_json(), to_binary(), or
         * to_binary() as a base64 encoded string.
         */
        explicit face_coordinates_with_classifiers(const json::value& val);
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
    json::value to_json(const face_coordinates_with_classifiers& fca);
    stdx::binary to_binary(const face_coordinates_with_classifiers& fca,
                           unsigned format = 0);
}
