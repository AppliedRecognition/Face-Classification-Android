
#include "prototype.hpp"
#include "internal_multiface.hpp"
#include "internal_serialize.hpp"
#include "model.hpp"

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <json/zlib.hpp>

#include <applog/core.hpp>


using namespace rec;

static const json::string K_proto("proto");
static const json::string K_uuid("uuid");

void prototype::set_comparison_variant(stdx::arg<core::context_data> context,
                                       version_type version,
                                       variant var) {
    if (!context)
        throw std::invalid_argument("context is null");
    if (auto p = core::get<context_map>(context->context).get(version)) {
        if (var == variant::none)
            var = p->default_compare_variant;
        p->compare_variant.store(var);
    }
}

void prototype::set_serialize_format(stdx::arg<core::context_data> context,
                                     version_type version,
                                     int format) {
    if (!context)
        throw std::invalid_argument("context is null");
    if (auto p = core::get<context_map>(context->context).get(version))
        p->serialize_format.store(format);
}

prototype_ptr
prototype::random(core::active_job job,
                  version_type ver) {
    auto& td = job.context().data;
    if (auto ms = core::get<context_map>(td.context).get(ver))
        if (ms->random)
            return ms->random(td,ms,nullptr,0,variant::none);
    FILE_LOG(logERROR) << "random prototype generation not implemented for version " << ver;
    throw std::runtime_error("not implemented");
}
prototype_ptr
prototype::random(core::active_job job,
                  stdx::arg<const prototype> base,
                  float score, variant var) {
    if (!base)
        throw std::invalid_argument("base prototype is null");
    const auto ver = base->version;
    auto& td = job.context().data;
    if (auto ms = core::get<context_map>(td.context).get(ver))
        if (ms->random)
            return ms->random(td,ms,base.get(),score,var);
    FILE_LOG(logERROR) << "random prototype generation not implemented for version " << ver;
    throw std::runtime_error("not implemented");
}

prototype_ptr
prototype::deserialize_bin(const core::context_data& context,
                           const void* src, std::size_t len,
                           const std::optional<uuid_type>& uuid) {
    if (!src || !len)
        throw std::invalid_argument("no data provided");
    const auto ver = *static_cast<const unsigned char*>(src);
    if (auto ms = core::get<context_map>(context.context).get(ver))
        if (ms->deserialize_prototype)
            if (auto p = ms->deserialize_prototype(ms, src, len, uuid))
                return p;
    FILE_LOG(logERROR) << "recognition engine deserialize prototype failed";
    throw std::runtime_error("recognition engine failure (prototype)");
}

prototype_ptr
prototype::deserialize(stdx::arg<const core::context_data> context,
                       const void* src, std::size_t len) {
    if (!context)
        throw std::invalid_argument("context is null");
    stdx::binary buf;
    while (internal::is_compressed(src,len)) {
        buf = internal::remove_compression(src,len);
        src = buf.data();
        len = buf.size();
    }
    if (internal::is_prototype(src,len))
        return deserialize_bin(*context,src,len);
    return deserialize(*context,json::decode_any(src,len));
}
prototype_ptr
prototype::deserialize(stdx::arg<const core::context_data> context,
                       const json::value& val) {
    if (!context)
        throw std::invalid_argument("context is null");
    if (!json::is_type<json::object>(val)) {
        auto bin = make_binary(val);
        return deserialize(*context, bin.data(), bin.size());
    }
    auto& obj = get_object(val);

    if (obj[K_proto] != json::null) {
        std::optional<uuid_type> uuid;
        if (obj[K_uuid] != json::null) {
            const auto bin = make_binary(obj[K_uuid]);
            if (bin.size() == uuid_bytes) {
                uuid.emplace();
                memcpy(uuid->data(), bin.data(), bin.size());
            }
            else
                FILE_LOG(logWARNING) << "prototype object has invalid uuid";
        }
        const auto bin = make_binary(obj[K_proto]);
        return deserialize_bin(*context,bin.data(),bin.size(),uuid);
    }
    
    const auto mf = internal::multiface::deserialize(*context, obj);
    auto vec = mf->get_prototypes();
    if (vec.size() != 1)
        throw std::invalid_argument("multiface does not have a single face");
    return move(vec.front());
}

prototype_ptr
rec::transcribe(stdx::arg<const core::context_data> context,
                stdx::arg<const prototype> a, version_type target_version) {
    if (!a) throw std::invalid_argument("null prototype argument");
    return a->version == target_version ?
        a->copy() : a->transcribe_to(*context, target_version);
}

compare_result rec::compare(stdx::arg<const prototype> a,
                            stdx::arg<const prototype> b,
                            variant var) {
    if (!a || !b)
        throw std::invalid_argument("null prototype argument");
    if (a->version != b->version)
        throw std::invalid_argument("prototype version mismatch");
    return a->compare_to(*b, var);
}

namespace rec {
    std::vector<float> to_float_vector(stdx::arg<const prototype> proto) {
        if (!proto) throw std::invalid_argument("null prototype argument");
        auto p = proto->vector_for_pca(0);
        auto last = p.first;
        advance(last, p.second);
        return { p.first, last };
    }

    json::value to_json(stdx::arg<const prototype> proto) {
        if (!proto) throw std::invalid_argument("null prototype argument");
        return json::object {
            { K_proto, proto->serialize() },
            { K_uuid,  proto->uuid }
        };
    }

    stdx::binary to_binary_with_opts(
        stdx::arg<const prototype> proto,
        stdx::options_tuple<serialize_type,compression_type> opts) {
        if (!proto) throw std::invalid_argument("null prototype argument");
        stdx::binary result;
        switch (std::get<serialize_type>(opts)) {
        case serialize_type::raw:
            result = proto->serialize();
            break;
            
        case serialize_type::json:
            result = stdx::binary(json::encode_json(to_json(*proto)));
            break;

        case serialize_type::cbor:
            result = json::encode_cbor(to_json(*proto));
            break;

        case serialize_type::def:
        case serialize_type::amf3:
            result = json::encode_amf3(to_json(*proto));
            break;
        }
        if (std::get<compression_type>(opts) == rec::deflate)
            result = json::pull_deflate(result).pull_final();
        return result;
    }
}
