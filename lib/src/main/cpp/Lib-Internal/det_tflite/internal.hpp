#pragma once

#include <det/internal.hpp>
#include <det/internal_landmarks.hpp>

#include <core/context.hpp>

#include <tensorflow/lite/model_builder.h>

namespace det {
    namespace internal {
        struct tflite_models_loader {
            models::loader_function loader;
        };
        inline auto& get_loader(const core::context_data& data) {
            if (auto ptr = core::cptr<tflite_models_loader>(data.context))
                return ptr->loader;
            return core::cget<models_loader>(data.context).loader;
        }

        template <unsigned DETVER>
        detector_factory_function tflite_factory(core::context_data& data);

        template <det::landmark_options>
        landmarks_factory_function tflite_factory(core::context_data& data);


        /** \brief Storage for any tflite flatbuffer model.
         *
         * The object is stored in the context state and must be treated
         * as const.  For models where a non-const version is to be used,
         * a per-thread copy must be made.
         *
         * Create a subclass for each distinct model.
         */
        struct tflite_model {
            stdx::binary bin;
            using FlatBufferModel = ::tflite::FlatBufferModel;
            std::unique_ptr<const FlatBufferModel> model;

            inline auto const& operator*() const { return model; }

            template <typename... Args>
            tflite_model(core::context_data& data, Args&&... args) {
                const auto& loader = get_loader(data);
                auto&& r = loader(models::format::tflite,
                                  std::forward<Args>(args)...);
                auto& vec = r.models;
                if (!vec.empty()) {
                    auto& var = vec.front();

                    if (auto p = std::get_if<models::istream_ptr>(&var)) {
                        if (auto s = p->get()) {
                            //if (!r.path.empty()) FILE_LOG(logINFO) << "model stream: " << r.path;
                            // read entire file into binary
                            // todo: would be better if it was memory mapped
                            std::vector<char> buf(
                                std::istreambuf_iterator<char>(*s), {});
                            bin = stdx::binary(move(buf));
                        }
                    }
                    else if (auto p = std::get_if<stdx::binary>(&var)) {
                        if (!p->empty()) {
                            //if (!r.path.empty()) FILE_LOG(logINFO) << "model buffer: " << r.path;
                            bin = *p;
                        }
                    }
                    if (!bin.empty()) {
                        model = FlatBufferModel::BuildFromBuffer(
                            bin.data<char>(), bin.size());
                        if (model)
                            return;
                    }
                }
                throw std::runtime_error("model not found or failed to load");
            }
        };
    }
}
