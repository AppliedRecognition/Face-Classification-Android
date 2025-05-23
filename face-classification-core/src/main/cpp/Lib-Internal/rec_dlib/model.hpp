#pragma once

#include <rec/model.hpp>
#include <models/types.hpp>
#include <dlibx/net_vector.hpp>

namespace rec::dlib {
    /** \brief Initialize for custom models.
     */
    constexpr void model_init() {}

    /** \brief Load stock or custom model from stream.
     */
    dlibx::net::vector model_load(std::istream&);

    /** \brief Attempt to load model as shared_ptr.
     * \returns nullptr on failure
     */
    std::shared_ptr<dlibx::net::vector>
    load_shared(version_type ver, const core::context_data& cd);

    /** \brief Loader for recognition models.
     */
    struct models_loader {
        models::loader_function loader;
    };

    /** \brief Per thread map of models.
     */
    class thread_map {
        std::map<version_type,
                 std::pair<dlibx::net::vector,
                           std::shared_ptr<const model_state> > > map;
    public:
        std::pair<dlibx::net::vector*, std::shared_ptr<const model_state> >
        get(version_type ver, const core::context_data& cd);
    };
}
