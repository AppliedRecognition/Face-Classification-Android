
#include "internal_multiface_2.hpp"
#include "internal_cluster_1.hpp"
#include "internal_prototype_1.hpp"
#include "internal_serialize.hpp"
#include "model.hpp"

#include <core/thread_data.hpp>

#include <applog/core.hpp>

#include <stdext/identity.hpp>


using namespace rec;
using namespace rec::internal;


namespace {
    const json::string K_cluster("cluster");
    const json::string K_clusters("clusters");
    const json::string K_cluster_threshold("cluster_threshold");
    const json::string K_face("face");
    const json::string K_faces("faces");
    const json::string K_ids("ids");
    const json::string K_proto("proto");
    const json::string K_pver("pver");
    const json::string K_uuid("uuid");
    const json::string K_ver("ver");
}

/* flattened:
 *
 * note: "clusters" must have exactly 1 entry
 *
 * v2 {
 *   "ver": 2,
 *   "pver": <int>,
 *   "cluster_threshold": 0,
 *   "clusters": [
 *     {
 *       "faces": [ { "uuid": <bin>, "face": ?<bin>, "ids": ?<array> }, ... ],
 *       "cluster": <bin>,
 *     },
 *     ...
 *   ]
 * }
 */

static auto decode_face(const core::context_data& cd, json::value src) {
    std::tuple<uuid_type,prototype_ptr,json::value> r;
    auto& uuid = std::get<uuid_type>(r);
    auto& dest = std::get<json::value>(r); 
    if (!json::is_type<json::object>(src)) {
        auto bin = make_binary(src);
        while (internal::is_compressed(bin.data(), bin.size()))
            bin = internal::remove_compression(bin.data(), bin.size());
        if (internal::is_prototype(bin.data(), bin.size())) {
            auto& p = std::get<prototype_ptr>(r) =
                prototype_1::deserialize(cd, bin.data(), bin.size());
            uuid = p->uuid;
            dest = bin;
            return r;
        }
        src = json::decode_any(bin);
    }
    auto& obj = get_object(src);
    const auto it = obj.find(K_uuid);
    if (it == obj.end())
        throw std::runtime_error("invalid multiface (uuid missing)");
    const auto bin = make_binary(it->second);
    if (bin.size() != uuid_bytes)
        throw std::runtime_error("invalid multiface (uuid invalid)");
    memcpy(uuid.data(), bin.data(), bin.size());
    dest = move(obj);
    return r;
}

multiface_2::~multiface_2() = default;

multiface_2::multiface_2(version_type ver, float) : multiface(ver) {}

multiface_2::multiface_2(const core::context_data& cd,
                         const json::object& top, face_map_type* face_map)
    : multiface(make_number(top[K_pver])) {

    if (get_integer(top[K_ver]) != 2)
        throw std::runtime_error("invalid multiface format");

    auto& c_arr = get_array(top[K_clusters]);
    if (c_arr.size() != 1)
        throw std::runtime_error("invalid multiface format");
    auto& c_obj = get_object(c_arr.front());
    
    cluster = std::make_unique<internal::cluster>(cd,c_obj[K_cluster]);
    if (cluster->model->version != version)
        throw std::runtime_error("invalid flattened multiface (cluster)");

    for (auto& f_obj : object_from_array(c_obj[K_faces])) {
        auto ids = get_array_safe(f_obj[K_ids]);
        if (f_obj[K_face] != json::null) {
            auto t = decode_face(cd, f_obj[K_face]);
            auto& uuid = std::get<uuid_type>(t);
            auto& proto = std::get<prototype_ptr>(t);
            if (proto && proto->version != version)
                throw std::runtime_error("invalid multiface (prototype)");
            if (face_map) {
                auto& r = (*face_map)[uuid];
                if (!ids.empty()) {
                    auto& v = std::get<json::array>(r);
                    if (!v.empty())
                        throw std::runtime_error(
                            "invalid multiface (duplicate uuid)");
                    v = move(ids);
                }
                std::get<json::value>(r) = std::get<json::value>(t);
            }
            proto_map.emplace(uuid, proto);
        }
        else if (f_obj[K_uuid] != json::null) {
            const auto bin = make_binary(f_obj[K_uuid]);
            if (bin.size() != uuid_bytes)
                throw std::runtime_error("invalid multiface (uuid)");
            uuid_type uuid;
            memcpy(uuid.data(), bin.data(), bin.size());
            if (face_map && !ids.empty()) {
                auto& v = std::get<json::array>((*face_map)[uuid]);
                if (!v.empty())
                    throw std::runtime_error(
                        "invalid multiface (duplicate uuid)");
                v = move(ids);
            }
            proto_map.emplace(uuid,nullptr);
        }
        else       
            throw std::runtime_error("invalid multiface (missing uuid)");
    }

    if (proto_map.size() == 1 && !proto_map.begin()->second) {
        if (auto proto = cluster->get_single_face()) {
            // special case -- ensure single prototype is in proto_map
            if (proto->version != version)
                throw std::runtime_error("invalid flattened multiface (face)");
            if (proto->uuid != proto_map.begin()->first) {
                FILE_LOG(logWARNING) << "prototype uuid inconsistency";
                proto = static_cast<const prototype_1*>(proto.get())
                    ->copy(proto_map.begin()->first);
                assert(proto->uuid == proto_map.begin()->first);
            }
            proto_map.begin()->second = proto;
        }
    }
}

json::value multiface_2::diagnostic() const {
    static const auto K_class = json::string("class");
    static const auto K_ver = json::string("ver");
    static const auto K_size = json::string("size");

    json::object top;
    top[K_class] = "multiface_2";
    top[K_ver] = version;
    top[K_size] = size();
    return top;
}

json::object
multiface_2::serialize(const face_map_type* face_map) const {

    json::array clusters;
    {
        if (proto_map.empty())
            throw std::logic_error("multiface has empty cluster");
        json::array faces;
        for (const auto& pr : proto_map) {
            bool need_uuid = true;
            json::object obj;
            if (face_map) {
                const auto it = face_map->find(pr.first);
                if (it != face_map->end()) {
                    const auto& v = std::get<json::value>(it->second);
                    if (v != json::null) {
                        obj[K_face] = v;
                        need_uuid = false;
                    }
                    const auto& ids = std::get<json::array>(it->second);
                    if (!ids.empty())
                        obj[K_ids] = ids;
                }
            }
            if (need_uuid)
                obj[K_uuid] = pr.first;
            faces.push_back(move(obj));
        }
        
        json::object obj;
        obj[K_faces] = move(faces);

        if (!cluster)
            throw std::logic_error("multiface has invalid cluster");
        const auto bin = cluster->serialize();
        if (bin.empty())
            throw std::logic_error("failed to serialize cluster");
        obj[K_cluster] = bin;

        clusters.push_back(move(obj));
    }

    json::object top;
    top[K_ver] = 2;
    top[K_pver] = version;
    top[K_clusters] = move(clusters);
    top[K_cluster_threshold] = 0.0f;
    return top;
}

void multiface_2::assign(stdx::forward_iterator<prototype_ptr> first,
                         stdx::forward_iterator<prototype_ptr> last) {

    proto_map_type new_map;
    for ( ; first != last ; ++first) {
        auto face = *first;
        if (!face || face->version != version || face->uuid.empty()) {
            FILE_LOG(logERROR) << "update_multiface: invalid prototype";
            throw std::invalid_argument("invalid prototype argument");
        }
        new_map.emplace(face->uuid, move(face));
    }

    // determine if there have been changes
    if (proto_map.size() == new_map.size()) {
        std::vector<uuid_type> a;
        a.reserve(proto_map.size());
        for (auto& p : proto_map)
            a.push_back(p.first);
        std::sort(a.begin(), a.end());
        std::vector<uuid_type> b;
        b.reserve(new_map.size());
        for (auto& p : new_map)
            b.push_back(p.first);
        std::sort(b.begin(), b.end());
        if (a == b) return;  // no changes
    }

    cluster.reset(
        new internal::cluster(
            { new_map.begin(), stdx::get_n<1>() },
            { new_map.end(),   stdx::get_n<1>() }));
    proto_map.swap(new_map);
}

std::size_t multiface_2::size() const {
    return proto_map.size();
}

uuid_set_type multiface_2::uuid_set() const {
    uuid_set_type result;
    result.reserve(proto_map.size());
    for (auto& p : proto_map)
        result.emplace_back(p.first);
    return result;
}

std::vector<prototype_ptr> multiface_2::get_prototypes() const {
    std::vector<prototype_ptr> result;
    result.reserve(proto_map.size());
    for (auto& p : proto_map)
        if (p.second) result.emplace_back(p.second);
    return result;
}

void multiface_2::compare_to_n(
    const prototype* const* protos, std::size_t n,
    variant var, float* results) const {

    assert(cluster);
    for ( ; n > 0; ++results, ++protos, --n)
        *results = cluster->compare_to(**protos, var);
    /*
    if (!(var & variant::remove_subject_bias))
        // todo: put subject bias in ? how ?
    */
}
