#pragma once

#include "internal_multiface.hpp"
#include "prototype.hpp"

#include <unordered_map>
#include <list>

namespace rec {
    namespace internal {

        struct cluster;
        
        /** \brief Multiface with exclusive inclusion clustering.
         *
         * With this form of clustering, for any cluster with 2 or more faces,
         * all pairs of faces within the cluster compare to within the
         * threshold -- they form a clique.
         *
         * Internal "ver": 3.
         */
        class multiface_3 final : public multiface {
        public:
            multiface_3(version_type ver, float threshold);
            multiface_3(const core::context_data& cd,
                        const json::object& obj, face_map_type* face_map);
            ~multiface_3();
            
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
            const float threshold;
            std::vector<internal::cluster> clusters;
        };
    }
}
