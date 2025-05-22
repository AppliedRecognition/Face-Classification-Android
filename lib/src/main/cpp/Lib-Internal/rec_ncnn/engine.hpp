#pragma once

#include <rec/types.hpp>
#include <models/loader.hpp>
#include <core/context.hpp>
#include <stdext/arg.hpp>
#include <istream>

namespace rec {
    struct model_static;

    namespace ncnn {
        /** \brief Register recognition engine with context.
         *
         * Note that models_loader is optional since models are only required
         * for prototype extraction.
         */
        void initialize(stdx::arg<core::context> context,
                        models::loader_function models_loader = {});

        /** \brief Register recognition engine with context.
         *
         * This version accepts a models directory path.
         */
        template <typename PATH>
        inline std::enable_if_t<stdx::is_path_v<PATH> >
        initialize(stdx::arg<core::context> context, PATH models_path) {
            initialize(context, models::loader<PATH>(std::move(models_path)));
        }
    }
}
