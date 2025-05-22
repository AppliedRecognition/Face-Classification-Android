#pragma once

#include <rec/types.hpp>
#include <models/loader.hpp>
#include <core/context.hpp>
#include <stdext/arg.hpp>
#include <istream>

namespace rec {
    struct model_static;

    namespace dlib {
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

        /** \brief Load a custom model file with a temporary version number.
         *
         * Must call initialize() first.
         *
         * If the version of an existing model is specified, then parameters
         * are copied from it.
         * Otherwise, a parameters object must be specified.
         * \sa model.hpp
         */
        version_type load_temporary(stdx::arg<core::context> context,
                                    stdx::arg<std::istream> model_stream,
                                    version_type parameters_from_version);

        version_type load_temporary(stdx::arg<core::context> context,
                                    stdx::arg<std::istream> model_stream,
                                    const model_static& parameters);

        template <typename PATH, typename PARAM>
        inline std::enable_if_t<stdx::is_path_v<PATH>,version_type>
        load_temporary(stdx::arg<core::context> context,
                       PATH model_filename,
                       const PARAM& param) {
            return load_temporary(
                context, models::open_binary_file(model_filename), param);
        }
    }
}
