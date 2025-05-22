#pragma once

#include "prototype.hpp"
#include "fpvc.hpp"
#include <raw_image/types.hpp>
#include <variant>

namespace rec {
    class engine;
    struct model_state;
    namespace internal {
        class multiface;
    }

    namespace internal {
        /** \brief Prototype for resnet and facenet recognition.
         *
         * One feature vector per face.
         * Vectors are serialized in accordance with the model's
         * serialize_format value (either 8, 12 or 16 bits per element).
         *
         * If model->cos_max_score > 0:
         *    cos_score = cos_max_score * normalized_inner_product
         *
         * If model->l2sqr_max_score > 0 and l2sqr_coeff > 0:
         *    l2sqr_score = l2sqr_max_score - l2sqr_coeff * square_distance
         */
        class prototype_1 : public prototype {
        public:
            const std::shared_ptr<const model_state> model;
            raw_image::plane_ptr thumb;  ///< extracted thumbnail

            static std::shared_ptr<prototype_1>
            make_shared(
                std::shared_ptr<const model_state> model,
                std::variant<internal::fpvc_vector_type,internal::fp16vec> vec,
                const std::optional<uuid_type>& uuid = {});

            static prototype_ptr
            deserialize(std::shared_ptr<const model_state> model,
                        const void* src, std::size_t len,
                        const std::optional<uuid_type>& uuid = {});

            static prototype_ptr
            deserialize(const core::context_data& cd,
                        const void* src, std::size_t len,
                        const std::optional<uuid_type>& uuid = {});

            static prototype_ptr
            random(core::thread_data& td,
                   std::shared_ptr<const model_state> model,
                   const prototype* base = nullptr,
                   float score = 0, variant var = variant::none);

            std::unique_ptr<internal::multiface>
            construct_multiface(float cluster_threshold) const override;

            raw_image::plane_ptr
            diagnostic_image(diagnostic, core::context_data*) const override;

            /// const access to data
            virtual std::pair<stdx::span<const uint8_t>,float> get8() const = 0;

            // tuple is { data, coeff, invnorm }
            virtual std::tuple<stdx::span<const int16_t>,float,float>
            get16() const = 0;

            /// with original norm
            inline auto get32_orig() const {
                auto t = get16();
                return std::pair<stdx::forward_iterator<float>,unsigned> {
                    { std::get<0>(t).begin(),
                      [c=std::get<1>(t)](auto x) { return c*x; } },
                    unsigned(std::get<0>(t).size())
                };
            }
            /// with unit norm
            inline auto get32_unit() const {
                auto t = get16();
                return std::pair<stdx::forward_iterator<float>,unsigned> {
                    { std::get<0>(t).begin(),
                      [c=std::get<2>(t)](auto x) { return c*x; } },
                    unsigned(std::get<0>(t).size())
                };
            }

            // make these methods public here
            using prototype::copy;
            using prototype::serialize;

        protected:
            prototype_1(std::shared_ptr<const model_state> model,
                        const uuid_type& uuid);
        };


        /// fixed size aligned storage for 16-bit prototypes
        template <unsigned VEC_SIZE>
        struct alignas(32) vec16_N {
            int16_t vec[VEC_SIZE];
            float coeff;
        };

        /** \brief Final object with storage of prototype data.
         *
         * The template argument VEC_SIZE is either 0 for any vector size,
         * or a fixed number (e.g. 128) for more efficient storage of
         * the vectors.
         */
        template <unsigned VEC_SIZE = 0>
        struct prototype_1_final final : public prototype_1 {
            // note: these pair<float,array<...> > types will have the
            // the array 32-bit aligned and padded to a multiple of 32-bits
            // in size
            using vec8_type = std::conditional_t<
                VEC_SIZE == 0, internal::fpvc_vector_type,
                std::pair<float, std::array<uint8_t,VEC_SIZE> > >;
            using vec16_type = std::conditional_t<
                VEC_SIZE == 0, internal::fp16vec, vec16_N<VEC_SIZE> >;

            const vec8_type vec8;
            const vec16_type vec16;
            const float invnorm16;  ///< = 1 / norm(vec16.values)

            std::pair<stdx::span<const uint8_t>,float>
            get8() const override {
                if (!(vec8.first > 0))
                    return { {}, 0.0f };
                return {
                    { vec8.second.data(), vec8.second.size() },
                    vec8.first
                };
            }

            std::tuple<stdx::span<const int16_t>,float,float>
            get16() const override;

            stdx::binary serialize() const override;

            std::pair<stdx::forward_iterator<float>,unsigned>
            vector_for_pca(unsigned i = 0) const override;

            prototype_ptr
            transcribe_to(const core::context_data& cd,
                          version_type target_version) const override;

            compare_result
            compare_to(const prototype& other, variant var) const override;

            prototype_ptr copy(const std::optional<uuid_type>& new_uuid) const override;

            prototype_1_final(
                std::shared_ptr<const model_state> model,
                std::variant<internal::fpvc_vector_type,internal::fp16vec> vec,
                const std::optional<uuid_type>& uuid = {});

            prototype_1_final(
                std::shared_ptr<const model_state> model,
                const vec8_type& vec8, const vec16_type& vec16,
                const uuid_type& uuid);

        private:
            vec16_type move_vec16(
                std::variant<internal::fpvc_vector_type,internal::fp16vec>& vec) const;
        };
    }
}
