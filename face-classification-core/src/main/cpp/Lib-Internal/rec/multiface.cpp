
#include "multiface.hpp"
#include "internal_multiface_2.hpp"
#include "internal_multiface_3.hpp"
#include "prototype.hpp"
#include "internal_serialize.hpp"

#include <json/zlib.hpp>

#include <applog/core.hpp>


using namespace rec;


static const json::string K_clusters("clusters");
static const json::string K_cluster_threshold("cluster_threshold");
static const json::string K_proto("proto");
static const json::string K_ver("ver");
static const json::string K_uuid("uuid");


/****************  class internal::multiface  ****************/

std::unique_ptr<internal::multiface>
internal::multiface::deserialize(const core::context_data& cd,
                                 const json::object& obj,
                                 face_map_type* face_map) {
    static const json::string K_ver("ver");
    switch (get_integer(obj[K_ver])) {
    case 2: return std::make_unique<internal::multiface_2>(cd,obj,face_map);
    case 3: return std::make_unique<internal::multiface_3>(cd,obj,face_map);
    }
    throw std::runtime_error("invalid multiface format");
}

void internal::multiface_deleter::operator()(const multiface* ptr) const {
    delete ptr;
}


/****************  class rec::multiface  ****************/

multiface::~multiface() = default;
multiface::multiface(multiface&&) = default;
multiface& multiface::operator=(multiface&&) = default;

multiface::multiface(float cluster_threshold)
    : cluster_threshold(cluster_threshold) {}

multiface::multiface(stdx::forward_iterator<prototype_ptr> first,
                     stdx::forward_iterator<prototype_ptr> last,
                     float cluster_threshold)
    : multiface(cluster_threshold) {
    assign(first, last);
}

multiface::multiface(
    stdx::arg<const core::context_data> context, json::value val)
    : cluster_threshold(0) {

    if (!context)
        throw std::invalid_argument("context is null");

    while (!json::is_type<json::object>(val)) {
        auto bin = make_binary(val);
        while (internal::is_compressed(bin.data(),bin.size()))
            bin = internal::remove_compression(bin.data(),bin.size());
        if (internal::is_prototype(bin.data(),bin.size())) {
            auto proto =
                prototype::deserialize_bin(*context,bin.data(),bin.size());
            state = proto->construct_multiface(cluster_threshold);
            state->assign(&proto, &proto + 1);
            return;
        }
        val = json::decode_any(bin);
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
        auto proto =
            prototype::deserialize_bin(*context,bin.data(),bin.size(),uuid);
        state = proto->construct_multiface(cluster_threshold);
        state->assign(&proto, &proto + 1);
        return;
    }

    cluster_threshold = make_number(obj[K_cluster_threshold]);
    if (get_integer_safe(obj[K_ver]) != 2 ||
        !json::is_type<json::array>(obj[K_clusters]) ||
        !get_array(obj[K_clusters]).empty())
        state = internal::multiface::deserialize(*context,obj);
}

void multiface::assign(stdx::forward_iterator<prototype_ptr> first,
                       stdx::forward_iterator<prototype_ptr> last) {
    if (first == last)
        state = nullptr;
    else {
        const auto v = (*first)->version;
        for (auto it = first; ++it != last; )
            if ((*it)->version != v)
                throw std::invalid_argument("prototype version mismatch");
        if (!state || state->version != v)
            state = (*first)->construct_multiface(cluster_threshold);
        state->assign(first, last);
    }
}

multiface::size_type multiface::size() const {
    return state ? state->size() : 0;
}

version_type multiface::version() const {
    return state ? state->version : 0;
}

multiface_ptr multiface::release() {
    if (!state) throw std::invalid_argument("multiface has no faces");
    return multiface_ptr(state.release());
}

namespace rec {
    json::value to_json(const multiface& mf) {
        json::object top;
        if (mf.state)
            top = mf.state->serialize();
        else {
            top[K_ver] = 2;
            top[K_clusters] = json::array();
        }
        if (top[K_cluster_threshold] == json::null)
            top[K_cluster_threshold] = mf.cluster_threshold;
        return top;
    }

    stdx::binary to_binary_with_opts(
        const multiface& mf,
        const stdx::options_tuple<serialize_type,compression_type>& opts) {
        return rec::to_binary_with_opts(to_json(mf), opts);
    }
}

compare_result rec::compare(
    stdx::arg<const internal::multiface> mf,
    stdx::arg<const prototype> proto, variant var) {
    if (!proto)
        throw std::invalid_argument("null prototype argument");
    if (!mf)
        throw std::invalid_argument("multiface has no faces");
    if (mf->version != proto->version)
        throw std::invalid_argument("multiface prototype version mismatch");
    float score;
    const prototype* p = proto.get();
    mf->compare_to_n(&p, 1, var, &score);
    return score;
}
