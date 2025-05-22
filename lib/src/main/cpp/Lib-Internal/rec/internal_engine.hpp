#pragma once

#include "types.hpp"
#include "internal_multiface.hpp"
#include <raw_image/types.hpp>
#include <json/types.hpp>
#include <stdext/span.hpp>
#include <stdext/forward_iterator.hpp>

namespace rec {
    using multi_plane_arg = stdx::spanarg<const raw_image::plane>;

    namespace internal {
        /** \brief Abstract base for per-implementation operations.
         */
        class engine {
        public:
            virtual ~engine() = default;

            virtual void load_model(const core::context_data& cd,
                                    version_type version) const = 0;
            
            virtual rotated_box
            bounding_box(
                const core::context_data& cd,
                const det::face_coordinates& rbox,
                version_type version) const = 0;

            virtual prototype_ptr
            extract_prototype(
                core::thread_data& td,
                const multi_plane_arg& image,
                const rotated_box& rbox,
                version_type version,
                const json::object& settings) const = 0;

            virtual prototype_ptr
            extract_prototype(
                core::thread_data& td,
                const multi_plane_arg& image,
                const det::face_coordinates& coordinates,
                version_type version,
                const json::object& settings) const = 0;

            virtual std::vector<prototype_ptr>
            extract_jitter(
                core::thread_data& td,
                const multi_plane_arg& image,
                const det::face_coordinates& coordinates,
                version_type version,
                const json::object&) const {
                return { extract_prototype(td,image,coordinates,version,{}) };
            }
        };
    }

    /** \brief Register an implementation.
     */
    void register_engine(core::context& context,
                         std::unique_ptr<internal::engine> ptr,
                         stdx::forward_iterator<version_type> first,
                         stdx::forward_iterator<version_type> last);
    inline void register_engine(core::context& context,
                                std::unique_ptr<internal::engine> ptr,
                                std::initializer_list<version_type> vers) {
        register_engine(context, move(ptr), std::begin(vers), std::end(vers));
    }

    /** \brief Register a temporary prototype version.
     *
     * The engine for this temporary version will be the same as for
     * the specified previously registered version.
     */
    version_type register_temporary(core::context& context, version_type ver);
}
