
#include "internal_engine.hpp"
#include "internal_serialize.hpp"
#include "model.hpp"
#include "prototype.hpp"

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <json/zlib.hpp>

#include <applog/core.hpp>

using namespace rec;


json::object internal::decode_object(json::value val) {
    while (!json::is_type<json::object>(val)) {
        auto bin = make_binary(val);
        while (internal::is_compressed(bin.data(),bin.size()))
            bin = internal::remove_compression(bin.data(),bin.size());
        if (internal::is_prototype(bin.data(),bin.size()))
            return { {"proto",bin} };
        val = json::decode_any(bin);
    }
    return get_object(val);
}

namespace {
    struct engine_tuple {
        version_type ver;
        bool loaded = false;
        const internal::engine* ptr = nullptr;
    };
    inline bool operator<(const engine_tuple& a, const engine_tuple& b) {
        return a.ver < b.ver;
    }
    struct rec_registration {
        std::vector<std::unique_ptr<internal::engine> > engine_list;
        std::vector<engine_tuple> engine_map;  ///< kept sorted
    };
}

static constexpr auto temporary_version_start = version_type(100);

void rec::register_engine(core::context& context,
                          std::unique_ptr<internal::engine> ptr,
                          stdx::forward_iterator<version_type> first,
                          stdx::forward_iterator<version_type> last) {
    const auto p = ptr.get();
    if (!p) throw std::invalid_argument("recognition engine pointer is null");
    auto& reg = core::emplace<rec_registration>(context.data().context);
    unsigned count = 0;
    for (auto it = first; it != last; ++it, ++count) {
        const auto v = *it;
        if (v <= 0 || temporary_version_start <= v) {
            FILE_LOG(logERROR) << "prototype version " << v << " not allowed";
            throw std::runtime_error("failed to register recognition engine");
        }
    }
    if (reg.engine_map.empty())
        reg.engine_map.reserve(count);
    reg.engine_list.emplace_back(move(ptr));
    for ( ; first != last; ++first)
        reg.engine_map.push_back({*first, false, p});
    stable_sort(reg.engine_map.begin(), reg.engine_map.end());
}

static const internal::engine&
find(version_type v, const core::context_data& data) {
    if (auto rp = core::ptr<rec_registration>(data.context)) {
        auto it = lower_bound(
            rp->engine_map.begin(), rp->engine_map.end(), engine_tuple{v});
        if (it == rp->engine_map.end() || it->ver != v)
            throw std::runtime_error("unknown prototype version");
        if (!it->loaded) {
            const auto end = upper_bound(
                rp->engine_map.begin(), rp->engine_map.end(), engine_tuple{v});
            auto jt = it;
            for (;;) {
                try {
                    assert(jt->ver == v && !jt->loaded);
                    jt->ptr->load_model(data,v);
                    break; // success
                }
                catch (const std::exception&) {
                }
                if (++jt == end) {
                    FILE_LOG(logERROR) << "failed to find a model for recognition version: " << v;
                    throw std::runtime_error("failed to find recognition model");
                }
            }
            if (it != jt)
                std::swap(*it,*jt);
            it->loaded = true;
            rp->engine_map.erase(next(it),end);
        }
        return *it->ptr;
    }
    throw std::runtime_error("recognition engine not available");
}

version_type rec::register_temporary(core::context& context, version_type v) {
    auto& e = find(v, context);
    if (auto rp = core::ptr<rec_registration>(context.data().context)) {
        const auto new_ver =
            std::max(rp->engine_map.back().ver + 1, temporary_version_start);
        rp->engine_map.push_back({new_ver, true, &e});
        return new_ver;
    }
    throw std::runtime_error("recognition engine not available");
}


/****************  namespace rec  ****************/

rotated_box
rec::bounding_box(stdx::arg<core::context_data> context,
                  const det::face_coordinates& coordinates,
                  version_type version) {
    if (!context) throw std::invalid_argument("context is nullptr");
    auto& e = find(version, *context);
    return e.bounding_box(*context, coordinates, version);
}

prototype_ptr
rec::extract(core::active_job job,
             const multi_plane_arg& image,
             const rotated_box& rbox,
             version_type version,
             const json::object& settings) {
    if (image.empty()) throw std::invalid_argument("image is empty");
    auto& e = find(version, job.context());
    if (auto p = e.extract_prototype(
            job.context(), image, rbox, version, settings))
        return p;
    FILE_LOG(logERROR) << "recognition engine extract failed";
    throw std::runtime_error("recognition engine failure (extract)");
}

prototype_ptr
rec::extract(core::active_job job,
             const multi_plane_arg& image,
             const det::face_coordinates& coordinates,
             version_type version,
             const json::object& settings) {
    if (image.empty()) throw std::invalid_argument("image is empty");
    auto& e = find(version, job.context());
    if (auto p = e.extract_prototype(
            job.context(), image, coordinates, version, settings))
        return p;
    FILE_LOG(logERROR) << "recognition engine extract failed";
    throw std::runtime_error("recognition engine failure (extract)");
}


/****************  class prototype  ****************/

void prototype::load_model(stdx::arg<core::context> context,
                           version_type version) {
    if (!context)
        throw std::invalid_argument("context is nullptr");
    find(version,*context);  // find does the load_model() call
}

std::vector<prototype_ptr>
prototype::jitter(core::active_job job,
                   const multi_plane_arg& image,
                   const det::face_coordinates& coordinates,
                   version_type version,
                   const json::object& options) {
    if (image.empty()) throw std::invalid_argument("image is empty");
    auto& e = find(version, job.context());
    auto v =
        e.extract_jitter(job.context(), image, coordinates, version, options);
    if (v.empty() || !v.front()) {
        FILE_LOG(logERROR) << "recognition engine extract failed";
        throw std::runtime_error("recognition engine failure (extract)");
    }
    return v;
}
