#pragma once

#include <det/detection.hpp>

namespace det {
    namespace ncnn {
        /** \brief Register detection engine with context.
         *
         * Note that models_loader is optional since it can also be specified
         * with det::set_models_loader().
         */
        void initialize(stdx::arg<core::context> context,
                        models::loader_function models_loader = {});

        /** \brief Register detection engine with context.
         *
         * This version accepts a models directory path.
         */
        template <typename PATH>
        inline std::enable_if_t<stdx::is_path_v<PATH> >
        initialize(stdx::arg<core::context> context, PATH models_path) {
            initialize(context, models::loader<PATH>(std::move(models_path)));
        }

        /// shortened name
        template <typename... Args>
        inline void init(Args&&... args) {
            initialize(std::forward<Args>(args)...);
        }
    }
}
