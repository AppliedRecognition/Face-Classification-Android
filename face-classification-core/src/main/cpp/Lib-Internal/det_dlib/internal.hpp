#pragma once

#include <det/internal.hpp>
#include <det/internal_landmarks.hpp>

#include <core/context.hpp>
#include <applog/core.hpp>

namespace det {
    namespace internal {
        struct dlib_models_loader {
            models::loader_function loader;
        };
        inline auto& get_loader(const core::context_data& data) {
            if (auto ptr = core::cptr<dlib_models_loader>(data.context))
                return ptr->loader;
            return core::cget<models_loader>(data.context).loader;
        }

        template <unsigned DETVER>
        detector_factory_function dlib_factory(core::context_data& data);

        template <det::landmark_options>
        landmarks_factory_function dlib_factory(core::context_data& data);

        /// Complete face detection with landmark detection.
        template <unsigned DETVER>
        struct dlib_job  {
            const detection_input& input;
            json::value* const diag;
            dlib_job(const detection_input& input,
                     json::value* diag = nullptr)
                : input(verify_no_rotation(input)), diag(diag) {}
            detection_result operator()(core::job_context&);
        };

        /** \brief Storage for any object that can be loaded deserialize().
         *
         * The object is stored in the context state and must be treated
         * as const.  For models where a non-const version is to be used,
         * a per-thread copy must be made.
         *
         * For distinct models of the same type, create subclasses.
         */
        template <typename T>
        struct dlib_object {
            using model_type = T;
            model_type model;

            inline T const& operator*() const { return model; }

            template <typename... Args>
            dlib_object(core::context_data& data, Args&&... args) {
                const auto& loader = get_loader(data);
                auto&& r = loader(models::format::dlib,
                                  std::forward<Args>(args)...);
                auto& vec = r.models;
                if (!vec.empty()) {
                    auto& var = vec.front();
                    if (auto p = std::get_if<models::istream_ptr>(&var)) {
                        if (auto s = p->get()) {
                            if (!r.path.empty())
                                FILE_LOG(logINFO) << "loading model: "
                                                  << r.path;
                            deserialize(model, *s);
                            return;
                        }
                    }
                    else if (auto p = std::get_if<stdx::binary>(&var)) {
                        if (!p->empty()) {
                            if (!r.path.empty())
                                FILE_LOG(logINFO) << "deserialize model: "
                                                  << r.path;
                            stdx::binarystream in(*p);
                            deserialize(model, in);
                            return;
                        }
                    }
                }
                throw std::runtime_error("model not found");
            }
        };
    }
}
