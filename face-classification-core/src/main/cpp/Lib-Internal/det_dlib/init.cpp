
#include "init.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <applog/core.hpp>

using namespace det::internal;

void det::dlib::initialize(stdx::arg<core::context> context,
                           models::loader_function loader) {
    if (!context) {
        FILE_LOG(logERROR) << "det::dlib::initialize: invalid context";
        throw std::invalid_argument("invalid context argument");
    }

    if (loader)
        core::emplace<const dlib_models_loader>(
            context->data().context, move(loader));

    insert_factory(*context,3,dlib_factory<3>(*context));
    insert_factory(*context,4,dlib_factory<4>(*context));
    insert_factory(*context,5,dlib_factory<5>(*context));
    insert_factory(*context,6,dlib_factory<6>(*context));
    insert_factory(*context,7,dlib_factory<7>(*context));

    insert_factory(*context,det::lm::dlib5,
                   dlib_factory<det::lm::dlib5>(*context));
    insert_factory(*context,det::lm::dlib68,
                   dlib_factory<det::lm::dlib68>(*context));
    insert_factory(*context,det::lm::mesh68,
                   dlib_factory<det::lm::mesh68>(*context));
    insert_factory(*context,det::lm::mesh478,
                   dlib_factory<det::lm::mesh478>(*context));
}
