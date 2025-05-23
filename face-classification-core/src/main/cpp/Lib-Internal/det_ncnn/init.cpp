
#include "init.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <applog/core.hpp>

using namespace det::internal;

void det::ncnn::initialize(stdx::arg<core::context> context,
                           models::loader_function loader) {
    if (!context) {
        FILE_LOG(logERROR) << "det::ncnn::initialize: invalid context";
        throw std::invalid_argument("invalid context argument");
    }

    if (loader)
        core::emplace<const ncnn_models_loader>(
            context->data().context, move(loader));

    insert_factory(*context,6,ncnn_factory<6>(*context));
    insert_factory(*context,7,ncnn_factory<7>(*context));

    insert_factory(*context,det::lm::mesh68,
                   ncnn_factory<det::lm::mesh68>(*context));
    insert_factory(*context,det::lm::mesh478,
                   ncnn_factory<det::lm::mesh478>(*context));
}
