#pragma once

#include "internal.hpp"
#include <raw_image/core.hpp>
#include <list>
#include <map>

namespace det {
    namespace internal {

        /** \brief Abstract base class for landmark detection.
         */
        class landmarks_base {
        public:
            virtual ~landmarks_base() = default;

            virtual detected_coordinates
            operator()(const detected_coordinates& dc,
                       const raw_image::plane& image,
                       core::thread_data& td,
                       unsigned contrast_correction) const = 0;
        };

        using landmarks_ptr = std::unique_ptr<landmarks_base>;

        using landmarks_factory_function = std::function<
            landmarks_ptr(core::context_data&, const landmark_settings&)>;

        void insert_factory(core::context_data& data,
                            det::landmark_options lmver,
                            landmarks_factory_function);

        std::vector<landmarks_base const*>
        load_landmark_detectors(
            core::context_data& data,
            const landmark_settings& settings);


        /** \brief Find specified landmarks.
         */
        struct landmark_detection_job {
            face_coordinates initial_position;
            const detection_input& input;
            std::vector<landmarks_base const*> detectors;
            unsigned idx;

            std::pair<unsigned, any_ptr> operator()(core::job_context&);
        };

        /** \brief Multiple landmark detection jobs.
         */
        struct landmark_jobs {
            using job_type =
                core::job_function<internal::landmark_detection_job>;
            std::list<job_type> job_list;
            std::multimap<unsigned, any_ptr> pending;
            unsigned expected_idx = 0;
            detection_result operator()(core::job_context& jc);
            friend void interrupt(landmark_jobs&);
        };
    }
}
