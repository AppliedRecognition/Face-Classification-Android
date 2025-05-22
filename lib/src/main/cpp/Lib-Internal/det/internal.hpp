#pragma once

// internal include file (do not include in files outside det library)
#ifndef __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#error det/internal.hpp is private (do not include)
#endif

#include "types.hpp"
#include "detection_internal.hpp"

#include <models/types.hpp>

#include <core/job_queue.hpp>
#include <core/thread_data.hpp>

#include <raw_image/core.hpp>
#include <raw_image/point_rounding.hpp>

#include <stdext/binarystream.hpp>

namespace det {
    namespace internal {
        struct models_loader {
            models::loader_function loader;
        };

        struct detection_input {
            raw_image::plane image;
            detection_settings settings;
            output_base const* output_constructor;
            bool low_latency;
        };

        struct detection_result {
            std::vector<any_ptr> faces;
            std::unique_ptr<core::job_result<detection_result> > next;
        };

        /** \brief Abstract base class for face detector.
         */
        class detector_base {
        public:
            virtual ~detector_base() = default;

            virtual void
            prepare_thread(core::job_context& jc,
                           const detection_settings& settings,
                           unsigned idx) = 0;

            virtual std::function<detection_result(core::job_context&)>
            detection_job(const detection_input& input,
                          json::value* diag = nullptr) const = 0;
        };

        using detector_ptr = std::unique_ptr<detector_base>;
        using detector_factory_function = std::function<
            detector_ptr(core::context_data&, const detection_settings&)>;

        void insert_factory(core::context_data& data,
                            unsigned detver,
                            detector_factory_function);

        detector_base& load_face_detector(
            core::context_data& data,
            const detection_settings& settings);


        detection_result
        landmark_detection(core::job_context& jc,
                           const detection_input& input,
                           std::vector<face_coordinates> faces);

        /// Throw exception if input is rotated (but mirror is ok).
        const detection_input& verify_no_rotation(const detection_input& input);


        /// Complete face detection with landmark detection.
        template <unsigned DETVER>
        struct detection_job  {
            const detection_input& input;
            json::value* const diag;
            detection_job(const detection_input& input,
                          json::value* diag = nullptr)
                : input(verify_no_rotation(input)), diag(diag) {}
            detection_result operator()(core::job_context&);
        };
    }
}
