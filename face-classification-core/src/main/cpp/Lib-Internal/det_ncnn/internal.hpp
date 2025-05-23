#pragma once

#include <det/internal.hpp>
#include <det/internal_landmarks.hpp>

#include <core/context.hpp>

namespace det {
    namespace internal {
        struct ncnn_models_loader {
            models::loader_function loader;
        };
        inline auto& get_loader(const core::context_data& data) {
            if (auto ptr = core::cptr<ncnn_models_loader>(data.context))
                return ptr->loader;
            return core::cget<models_loader>(data.context).loader;
        }

        template <unsigned DETVER>
        detector_factory_function ncnn_factory(core::context_data& data);

        template <det::landmark_options>
        landmarks_factory_function ncnn_factory(core::context_data& data);
    }
}
