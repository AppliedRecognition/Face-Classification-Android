#pragma once

#include "internal_multiface.hpp"
#include "prototype.hpp"

#include <unordered_map>
#include <list>

namespace rec {
    namespace internal {

        struct cluster;
        
        /** \brief Multiface with no clustering.
         *
         * All faces go into a single cluster object.
         *
         * Internal "ver": 2.
         */
        class multiface_2 final : public multiface {
          public:
            multiface_2(version_type ver, float);
            multiface_2(const core::context_data& cd,
                        const json::object& obj, face_map_type* face_map);
            ~multiface_2();
            
            void assign(
                stdx::forward_iterator<prototype_ptr> first,
                stdx::forward_iterator<prototype_ptr> last) override;
            
            std::size_t size() const override;
            uuid_set_type uuid_set() const override;

            std::vector<prototype_ptr> get_prototypes() const override;

            json::object serialize(
                const face_map_type* face_map) const override;

            void compare_to_n(
                const prototype* const* p, std::size_t n,
                variant var, float* results) const override;

            json::value diagnostic() const override;
            
          private:
            using proto_map_type = std::unordered_map<
                uuid_type, prototype_ptr, hash_from_uuid>;
            proto_map_type proto_map;
            std::unique_ptr<internal::cluster> cluster;
        };
    }
}
