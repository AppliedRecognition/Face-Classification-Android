
#include "engine.hpp"
#include "internal.hpp"
#include "model.hpp"
#include "extract.hpp"

#include <rec/prototype.hpp>

#include <core/context.hpp>
#include <core/thread_data.hpp>

#include <applog/core.hpp>

using rec::prototype_ptr;
namespace rd = rec::dlib;

void rd::initialize(stdx::arg<core::context> context,
                    models::loader_function loader) {
    model_init();
    if (!context)
        throw std::invalid_argument("context is nullptr");
    if (loader)
        core::emplace<const models_loader>(context->data().context, move(loader));
    core::emplace<context_map>(context->data().context);
    const auto v = context_map::known_versions();
    register_engine(
        *context, std::make_unique<dlib::engine>(), v.begin(), v.end());
}

rec::version_type
rd::load_temporary(stdx::arg<core::context> context,
                   stdx::arg<std::istream> in, const model_static& model) {
    if (!context)
        throw std::invalid_argument("context is nullptr");
    if (!in)
        throw std::invalid_argument("input stream is nullptr");
    auto nvp = std::make_shared<dlibx::net::vector>();
    try {
        *nvp = rec::dlib::model_load(*in);
    }
    catch (const std::exception& e) {
        FILE_LOG(logERROR) << "failed to load recognition model: " << e.what();
        throw std::runtime_error("failed to load recognition model");
    }
    const auto v = context_map::known_versions();
    assert(!v.empty());
    const auto ver = register_temporary(*context, v.front());
    core::get<context_map>(context->data().context).insert( {
            ver, model.default_compare_variant,
            model.cos_max_score,
            model.l2sqr_max_score, model.l2sqr_coeff, nullptr },
        move(nvp));
    return ver;
}

rec::version_type
rd::load_temporary(stdx::arg<core::context> context,
                   stdx::arg<std::istream> in, version_type ver) {
    if (!context)
        throw std::invalid_argument("context is nullptr");
    return load_temporary(
        context, in, *core::get<context_map>(context->data().context).get(ver));
}

void rd::engine::load_model(const core::context_data& cd, version_type ver) const {
    if (!core::get<context_map>(cd.context).load(
            ver, &load_shared, ver, cd).first)
        throw std::runtime_error("failed to load model");
}

rec::rotated_box
rd::engine::bounding_box(
    const core::context_data& cd,
    const det::face_coordinates& coordinates,
    version_type version) const {
    return rec::dlib::bounding_box(coordinates, version, cd);
}

prototype_ptr
rd::engine::extract_prototype(
    core::thread_data& td,
    const multi_plane_arg& image,
    const rotated_box& rbox,
    version_type version,
    const json::object& options) const {
    return extract(image, rbox, version, options, td);
}

prototype_ptr
rd::engine::extract_prototype(
    core::thread_data& td,
    const multi_plane_arg& image,
    const det::face_coordinates& coordinates,
    version_type version,
    const json::object& options) const {
    return extract(image, coordinates, version, options, td);
}

std::vector<prototype_ptr>
rd::engine::extract_jitter(
    core::thread_data& td,
    const multi_plane_arg& image,
    const det::face_coordinates& coordinates,
    version_type version,
    const json::object& options) const {
    return jitter(image, coordinates, version, options, td);
}
