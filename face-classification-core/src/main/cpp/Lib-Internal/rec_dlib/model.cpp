
#include "model.hpp"

#include <core/context.hpp>
#include <core/thread_data.hpp>

#include <applog/core.hpp>

#include <stdext/binarystream.hpp>


using rec::model_state;
using rec::dlib::thread_map;


/****************  class thread_map  ****************/

std::shared_ptr<dlibx::net::vector>
rec::dlib::load_shared(version_type ver, const core::context_data& cd) {
    const auto lptr = core::cptr<models_loader>(cd.context);
    if (!lptr) {
        FILE_LOG(logWARNING) << "models basepath not set for rec_dlib";
        return nullptr;
    }
    const auto& loader = lptr->loader;
    auto&& r = loader(models::format::dlib,
                      models::type::face_recognition,
                      models::face_recognition(ver));
    auto& vec = r.models;
    if (vec.empty()) {
        FILE_LOG(logWARNING) << "failed to find dlib recognition model: " << ver;
        return nullptr;
    }
    try {
        auto& var = vec.front();
        if (auto p = std::get_if<models::istream_ptr>(&var)) {
            if (auto s = p->get()) {
                FILE_LOG(logINFO) << "load[" << ver << "]: "
                                  << (r.path.empty() ?
                                      "(recognition model)" : r.path);
                return std::make_shared<dlibx::net::vector>(
                    rec::dlib::model_load(*s));
            }
        }
        else if (auto p = std::get_if<stdx::binary>(&var)) {
            if (!p->empty()) {
                FILE_LOG(logINFO) << "load[" << ver << "]: "
                                  << (r.path.empty() ?
                                      "(recognition model)" : r.path);
                stdx::binarystream in(*p);
                return std::make_shared<dlibx::net::vector>(
                    rec::dlib::model_load(in));
            }
        }
    }
    catch (const std::exception& e) {
        FILE_LOG(logERROR) << "failed to load recognition model: "
                           << e.what();
    }
    return nullptr;
}

std::pair<dlibx::net::vector*, std::shared_ptr<const model_state> >
thread_map::get(version_type ver, const core::context_data& cd) {
    auto& p = map[ver];
    if (p.first.empty()) {
        auto r = core::emplace<context_map>(cd.context)
            .load(ver,&load_shared,ver,cd);
        p.second = move(r.second);
        FILE_LOG(logDETAIL) << "thread_map: copy " << ver << ' '
                            << static_cast<const void*>(r.first.get());
        p.first = *r.first;
    }
    return std::make_pair(&p.first, p.second);
}

