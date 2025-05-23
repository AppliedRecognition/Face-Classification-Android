
#include "init.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <applog/core.hpp>

using namespace det::internal;

void det::tflite::initialize(stdx::arg<core::context> context,
                           models::loader_function loader) {
    if (!context) {
        FILE_LOG(logERROR) << "det::tflite::initialize: invalid context";
        throw std::invalid_argument("invalid context argument");
    }

    if (loader)
        core::emplace<const tflite_models_loader>(
            context->data().context, move(loader));

    insert_factory(*context,8,tflite_factory<8>(*context));

    insert_factory(*context,det::lm::mesh478,
                   tflite_factory<det::lm::mesh478>(*context));
}
