#pragma once

#include <rec/internal_engine.hpp>

namespace rec {
    namespace dlib {
        class engine : public internal::engine {
            void load_model(const core::context_data& context,
                            version_type version) const override;

            rotated_box
            bounding_box(
                const core::context_data& cd,
                const det::face_coordinates& rbox,
                version_type version) const override;

            prototype_ptr
            extract_prototype(
                core::thread_data& td,
                const multi_plane_arg& image,
                const rotated_box& coordinates,
                version_type version,
                const json::object& settings) const override;

            prototype_ptr
            extract_prototype(
                core::thread_data& td,
                const multi_plane_arg& image,
                const det::face_coordinates& coordinates,
                version_type version,
                const json::object& settings) const override;

            std::vector<prototype_ptr>
            extract_jitter(
                core::thread_data& td,
                const multi_plane_arg& image,
                const det::face_coordinates& coordinates,
                version_type version,
                const json::object& options) const override;
        };
    }
}
