#pragma once

#include <rec/types.hpp>
#include <raw_image/core.hpp>
#include <json/types.hpp>

namespace rec {
    namespace dlib {

        /** \brief Extract face.
         *
         * Uses dlib recognition method to extract template.
         */
        rotated_box bounding_box(
            const det::face_coordinates& coordinates,
            version_type version,
            const core::context_data& cd);
        prototype_ptr extract(
            const raw_image::multi_plane_arg& image,
            const rotated_box& rbox,
            version_type version,
            const json::object& options,
            core::thread_data& td);
        prototype_ptr extract(
            const raw_image::multi_plane_arg& image,
            const det::face_coordinates& coordinates,
            version_type version,
            const json::object& options,
            core::thread_data& td);

        /** \brief Extract jittered set of prototypes.
         */
        std::vector<prototype_ptr> jitter(
            const raw_image::multi_plane_arg& image,
            const det::face_coordinates& coordinates,
            version_type version,
            const json::object& options,
            core::thread_data& td);

        /** \brief Extract prototype from face chip.
         *
         * The diagnostic::extracted image from one prototype maybe used
         * (as is or altered) to create another prototype.
         */
        prototype_ptr from_face_chip(
            raw_image::plane_ptr face_chip,
            version_type version,
            core::thread_data& td);

        // note: reconstruction not possible from dlib template
    }
}
