
#include "internal_multiface_3.hpp"
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
 * v3 {
 *   "ver": 3,
 *   "pver": <int>,
 *   "cluster_threshold": <float>,
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

multiface_3::~multiface_3() = default;

multiface_3::multiface_3(version_type ver, float threshold)
    : multiface(ver), threshold(threshold) {}

multiface_3::multiface_3(const core::context_data& cd,
                         const json::object& top, face_map_type* face_map)
    : multiface(make_number(top[K_pver])),
      threshold(make_number(top[K_cluster_threshold])) {

    if (get_integer(top[K_ver]) != 3)
        throw std::runtime_error("invalid multiface format");

    auto& c_arr = get_array(top[K_clusters]);
    if (c_arr.empty())
        throw std::runtime_error("invalid multiface format (empty)");
    clusters.reserve(c_arr.size());

    for (auto& c_obj : object_from_array(c_arr)) {
        clusters.emplace_back(cd,c_obj[K_cluster]);
        auto& cluster = clusters.back();
        if (cluster.model->version != version)
            throw std::runtime_error("invalid flattened multiface (cluster)");

        if (face_map) {
            auto& f_arr = get_array(c_obj[K_faces]);
            if (f_arr.size() != cluster.size())
                throw std::runtime_error("invalid flattened multiface (cluster)");
            for (auto& f_obj : object_from_array(f_arr)) {
                auto ids = get_array_safe(f_obj[K_ids]);
                if (f_obj[K_face] != json::null) {
                    auto t = decode_face(cd, f_obj[K_face]);
                    auto& uuid = std::get<uuid_type>(t);
                    auto& proto = std::get<prototype_ptr>(t);
                    if (proto && proto->version != version)
                        throw std::runtime_error("invalid multiface (prototype)");
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
                else if (!ids.empty() && f_obj[K_uuid] != json::null) {
                    const auto bin = make_binary(f_obj[K_uuid]);
                    if (bin.size() != uuid_bytes)
                        throw std::runtime_error("invalid multiface (uuid)");
                    uuid_type uuid;
                    memcpy(uuid.data(), bin.data(), bin.size());
                    auto& v = std::get<json::array>((*face_map)[uuid]);
                    if (!v.empty())
                        throw std::runtime_error(
                            "invalid multiface (duplicate uuid)");
                    v = move(ids);
                }
                else       
                    throw std::runtime_error("invalid multiface (missing uuid)");
            }
        }
    }
}

json::object
multiface_3::serialize(const face_map_type* face_map) const {

    json::array clusters;
    for (auto& c : this->clusters) {
        if (c.faces.empty())
            throw std::logic_error("multiface has empty cluster");
        json::array faces;
        for (const auto& p : c.faces) {
            assert(p);
            bool need_uuid = true;
            json::object obj;
            if (face_map) {
                const auto it = face_map->find(p->uuid);
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
                obj[K_uuid] = p->uuid;
            faces.push_back(move(obj));
        }
        
        json::object obj;
        obj[K_faces] = move(faces);

        const auto bin = c.serialize();
        if (bin.empty())
            throw std::logic_error("failed to serialize cluster");
        obj[K_cluster] = bin;

        clusters.push_back(move(obj));
    }

    json::object top;
    top[K_ver] = 3;
    top[K_pver] = version;
    top[K_clusters] = move(clusters);
    top[K_cluster_threshold] = threshold;
    return top;
}

void multiface_3::assign(stdx::forward_iterator<prototype_ptr> first,
                         stdx::forward_iterator<prototype_ptr> last) {

    std::vector<prototype_ptr> protos;
    protos.reserve(size());
    for ( ; first != last ; ++first) {
        auto face = *first;
        if (!face || face->version != version || face->uuid.empty()) {
            FILE_LOG(logERROR) << "update_multiface: invalid prototype";
            throw std::invalid_argument("invalid prototype argument");
        }
        protos.emplace_back(move(face));
    }
    if (protos.empty())
        throw std::invalid_argument("multiface must have at least one face");

    {
        // determine if there have been changes
        uuid_set_type a;
        a.reserve(protos.size());
        for (auto& p : protos)
            a.emplace_back(p->uuid);
        std::sort(a.begin(), a.end());
        
        auto b = uuid_set();
        std::sort(b.begin(), b.end());
        if (a == b)
            return;  // nothing to do
    }
    
    // compute pair-wise scores
    const auto N = protos.size()*(protos.size()-1)/2;
    FILE_LOG(logDETAIL) << "multiface_3: doing " << N << " comparisons of "
                        << protos.size() << " faces";
    std::vector<bool> compatible(N, false);
    auto pmj = compatible.begin();
    std::vector<std::tuple<float,unsigned,unsigned> > scores;
    scores.reserve(N);
    for (unsigned i = 1; i < protos.size(); ++i) {
        auto& pi = *protos[i];
        for (unsigned j = 0; j < i; ++j, ++pmj) {
            const auto s = compare(pi, protos[j]);
            if (s >= threshold) {
                scores.emplace_back(s,i,j);
                *pmj = true;
            }
        }
    }

    // clusters
    struct cluster_rec {
        cluster_rec* leader;
        std::list<unsigned> els;
        cluster_rec() : leader(this) {}
        auto get_leader() const {
            auto r = leader;
            while (r != r->leader)
                r = r->leader;
            return r;
        }
    };
    std::vector<cluster_rec> clusters(protos.size());
    for (unsigned i = 0; i < protos.size(); ++i)
        clusters[i].els.push_back(i);

    // form clusters
    FILE_LOG(logDETAIL) << "multiface_3: sort";
    std::sort(scores.begin(), scores.end(), std::greater<>{});
    FILE_LOG(logDETAIL) << "multiface_3: clustering with threshold "
                        << threshold;
    for (auto& t : scores) {
        auto* ci = clusters[std::get<1>(t)].get_leader();
        auto* cj = clusters[std::get<2>(t)].get_leader();
        if (ci != cj && [&] {
                // figure out if we can cluster
                for (auto i : ci->els)
                    for (auto j : cj->els) {
                        auto k = i < j ? (j*(j-1))/2 + i : (i*(i-1))/2 + j;
                        if (!compatible[k])
                            return false;
                    }
                return true;
            }()) {
            // merge clusters (note: j < i)
            cj->els.splice(cj->els.begin(), move(ci->els));
            ci->leader = cj;
        }
    }
    const auto num_clusters =
        std::count_if(clusters.begin(), clusters.end(),
                      [](const auto& c) { return c.leader == &c; });
    FILE_LOG(logDETAIL) << "multiface_3: " << num_clusters << " clusters";

    // create final clusters
    std::vector<internal::cluster> new_clusters;
    new_clusters.reserve(std::size_t(num_clusters));
    std::vector<prototype_ptr> ps;
    ps.reserve(protos.size());
    for (auto& c : clusters) {
        if (c.leader == &c) {
            assert(!c.els.empty());
            for (auto i : c.els) {
                auto p = move(protos[i]);
                assert(p);
                ps.emplace_back(move(p));
            }
            new_clusters.emplace_back(ps.begin(),ps.end());
            ps.clear();
        }
    }
    this->clusters.swap(new_clusters);
    FILE_LOG(logDETAIL) << "multiface_3: done";
}

json::value multiface_3::diagnostic() const {
    static const auto K_class = json::string("class");
    static const auto K_ver = json::string("ver");
    static const auto K_size = json::string("size");
    static const auto K_num_clusters = json::string("num_clusters");
    static const auto K_threshold = json::string("threshold");

    json::object top;
    top[K_class] = "multiface_3";
    top[K_ver] = version;
    top[K_size] = size();
    top[K_num_clusters] = clusters.size();
    top[K_threshold] = threshold;
    return top;
}

std::size_t multiface_3::size() const {
    std::size_t n = 0;
    for (auto& c : clusters)
        n += c.size();
    return n;
}

uuid_set_type multiface_3::uuid_set() const {
    uuid_set_type result;
    result.reserve(size());
    for (auto& c : clusters)
        for (auto& p : c.faces) {
            assert(p);
            result.emplace_back(p->uuid);
        }
    return result;
}

std::vector<prototype_ptr> multiface_3::get_prototypes() const {
    std::vector<prototype_ptr> result;
    result.reserve(size());
    for (auto& c : clusters)
        for (auto& p : c.faces)
            result.emplace_back(p);
    return result;
}

void multiface_3::compare_to_n(
    const prototype* const* protos, std::size_t n,
    variant var, float* results) const {

    assert(!clusters.empty());
    for ( ; n > 0; ++results, ++protos, --n) {
        *results = -1e10;
        for (auto& c : clusters)
            *results = std::max(*results, c.compare_to(**protos, var));
        //if (!(var & variant::remove_subject_bias))
          // todo: put subject bias in ? how ?
    }
}
