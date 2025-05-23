#pragma once

#include <core/context.hpp>
#include <stdext/forward_iterator.hpp>

namespace json {
    class value;
}

namespace det {
    struct image_struct;

    namespace internal {
        using any_ptr = std::unique_ptr<void,void(*)(void*)>;
        
        struct output_base {
            virtual ~output_base() = default;

            virtual std::unique_ptr<output_base>
            copy(const face_coordinates&, core::job_context&) const = 0;

            virtual any_ptr
            operator()(face_coordinates&, core::job_context&) = 0;
        };

        template <typename FN>
        using output_type =
            std::decay_t<std::invoke_result_t<
                             FN&,face_coordinates&,core::job_context&> >;

        template <typename FN>
        struct output_fn final : output_base {
            using result_type = output_type<FN>;

            static void deleter(void* p) {
                delete static_cast<result_type*>(p);
            }

            FN fn;
            
            template <typename... Args>
            output_fn(Args&&... args) : fn(std::forward<Args>(args)...) {}

            std::unique_ptr<output_base>
            copy(const face_coordinates& face,
                 core::job_context& jc) const override {
                return std::make_unique<output_fn>(fn, face, jc);
            }

            any_ptr operator()(
                face_coordinates& face, core::job_context& jc) override {
                return { new result_type(fn(face,jc)), &deleter };
            }
        };

        struct detection_state;
        struct detection_state_deleter {
            void operator()(detection_state*);
        };
        using detection_state_ptr =
            std::unique_ptr<detection_state, detection_state_deleter>;

        detection_state_ptr start_detect_faces(
            core::active_job& context,
            const detection_settings& settings,
            const image_struct* image,
            std::unique_ptr<output_base> output_constructor,
            bool low_latency, json::value* diagnostic);

        detection_state_ptr start_detect_landmarks(
            core::active_job& context,
            const landmark_settings& landmarks,
            const image_struct* image,
            stdx::forward_iterator<const detected_coordinates&> first, 
            stdx::forward_iterator<const detected_coordinates&> last,
            std::unique_ptr<output_base> output_constructor);

        std::vector<any_ptr> get_some(detection_state&);
    }
}
