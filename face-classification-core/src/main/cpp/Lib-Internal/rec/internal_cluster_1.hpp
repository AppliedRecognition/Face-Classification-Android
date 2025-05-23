#pragma once

#include "prototype.hpp"
#include <vector>

namespace json {
    class value;
}

namespace rec {
    struct model_state;

    namespace internal {

        /** \brief Cluster of prototypes.
         *
         * Single prototype to multi-prototype cluster comparisons:
         *
         * variant::cos
         *   first compute mean face-to-face comparison score,
         *   then correct by scaling by 1 / norm(center_vector).
         *
         * variant::l2sqr
         *   compare single prototype to mean vector
         */
        struct cluster {
            const std::shared_ptr<const model_state> model;

            const std::vector<prototype_ptr> faces;
            const std::vector<float> mean_vec;
            const float cos_boost;

            cluster(const stdx::forward_iterator<prototype_ptr>& first,
                    const stdx::forward_iterator<prototype_ptr>& last);

            // deserialize from binary, base64 or array of binaries
            cluster(const core::context_data& cd, const json::value& v);

            /** \brief Serialize to array of binaries.
             */
            json::array serialize() const;

            /** \brief Number of faces represented by cluster.
             *
             * In some cases this value may be less than the number of faces
             * used to create the cluster.  
             * This can be due to similarity between faces or what is
             * effectively lossy compression during cluster creation.
             *
             * The return value will be >= 1.
             */
            inline auto size() const {
                return faces.size();
            }

            /** \brief Compare to prototype and return score.
             *
             * Variant constants are defined in class prototype.
             */
            float compare_to(
                const prototype& other, variant var = variant::none) const;

            /** \brief Diagnostic information.
             */
            json::value diagnostic() const;

            /** \brief Get face from single face cluster.
             */
            prototype_ptr get_single_face() const {
                return faces.size() == 1 ? faces.front() : nullptr;
            }
        };
    }
}
