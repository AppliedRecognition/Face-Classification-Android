#pragma once

#include <net.h>
#include <rec/model.hpp>
#include <models/types.hpp>

namespace raw_image {
    class input_extractor;
}

namespace rec::ncnn {
    /** \brief Known models and the input extractor required for them.
     */
    constexpr std::pair<version_type,std::string_view> known_models[] = {
        { 24u, "retina112*2.85+0.35rgb" }
    };

    /** \brief Loader for recognition models.
     */
    struct models_loader {
        models::loader_function loader;
    };

    /** \brief ncnn neural net and input_extractor.
     */
    struct model_record {
        ::ncnn::Net net;
        raw_image::input_extractor const* extractor = nullptr;
    };

    /** \brief Attempt to load model as shared_ptr.
     * \returns nullptr on failure
     */
    std::shared_ptr<model_record>
    load_shared(version_type ver, const core::context_data& cd);
}
