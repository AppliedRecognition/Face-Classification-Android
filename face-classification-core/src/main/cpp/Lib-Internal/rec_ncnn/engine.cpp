
#include "models.hpp"
#include "engine.hpp"
#include "internal.hpp"
#include "extract.hpp"

#include <rec/prototype.hpp>

#include <core/context.hpp>
#include <core/thread_data.hpp>

#include <applog/core.hpp>

#include <stdext/identity.hpp>

using namespace rec;
namespace rn = rec::ncnn;

void rn::initialize(stdx::arg<core::context> context,
                    models::loader_function loader) {
    if (!context)
        throw std::invalid_argument("context is nullptr");
    if (loader)
        core::emplace<const models_loader>(context->data().context, move(loader));
    core::emplace<context_map>(context->data().context);
    register_engine(
        *context, std::make_unique<ncnn::engine>(),
        { std::begin(known_models), stdx::get_n<0>{} },
        { std::end(known_models),   stdx::get_n<0>{} } );
}

void rn::engine::load_model(const core::context_data& cd, version_type ver) const {
    if (!core::get<context_map>(cd.context).load(
            ver, &load_shared, ver, cd).first)
        throw std::runtime_error("failed to load ncnn recognition model");
}

rotated_box
rn::engine::bounding_box(
    const core::context_data& cd,
    const det::face_coordinates& coordinates,
    version_type version) const {
    return rec::ncnn::bounding_box(coordinates, version, cd);
}

prototype_ptr
rn::engine::extract_prototype(
    core::thread_data& td,
    const multi_plane_arg& image,
    const rotated_box& rbox,
    version_type version,
    const json::object&) const {
    return extract(image, rbox, version, td);
}

prototype_ptr
rn::engine::extract_prototype(
    core::thread_data& td,
    const multi_plane_arg& image,
    const det::face_coordinates& coordinates,
    version_type version,
    const json::object&) const {
    return extract(image, coordinates, version, td);
}
