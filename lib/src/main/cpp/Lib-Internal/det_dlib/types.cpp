
#include "classifiers.hpp"

#include <dlibx/net_vector.hpp>

#include <json/types.hpp>
#include <json/zlib.hpp>

#include <mutex>

using namespace det;

static const auto K_n = json::string("n");
static const auto K_v = json::string("v");
static const auto K_fcver = json::string("fcver");
static const auto K_attr = json::string("attr");
static const auto K_det = json::string("det");

json::value det::to_json(const face_coordinates_with_classifiers& fca) {
    json::object obj;
    obj[K_fcver] = 1;
    obj[K_det] = to_json(static_cast<const face_coordinates&>(fca));
    if (!fca.classifiers.empty()) {
        json::array attr;
        attr.reserve(fca.classifiers.size());
        for (auto& p : fca.classifiers) {
            json::object o;
            o[K_n] = p.first->name;
            if (p.second.size() >= 2)
                o[K_v] = p.second;
            else if (p.second.size() == 1)
                o[K_v] = p.second.front();
            attr.push_back(move(o));
        }
        obj[K_attr] = move(attr);
    }
    return obj;
}

stdx::binary
det::to_binary(const face_coordinates_with_classifiers& fca, unsigned format) {
    const auto top = to_json(fca);

    stdx::binary r;
    if (format&2)
        r.assign(encode_json(top));
    else
        r = encode_amf3(top);

    if ((format&1) == 0)
        r = json::pull_deflate(r).pull_final();

    return r;
}

static auto decode(face_coordinates& fc, const json::value& top) {
    json::object const* obj = nullptr;
    const auto& arr = [&]() -> const json::array& {
        if (json::is_type<json::array>(top))
            return get_array(top);
        obj = &get_object(top);
        if (get_integer_safe((*obj)[K_fcver]) != 1)
            throw std::runtime_error(
                "invalid face_coordinates serialization (unknown version)");
        return get_array((*obj)[K_det]);
    }();
    fc.reserve(arr.size());
    for (auto& v : arr)
        fc.emplace_back(v);
    return obj;
}

static classifier_model_type const* empty_model(const json::string& s) {
    static const dlibx::net::vector empty_model;
    static const classifier_model_type empty_rec{{},{},empty_model};
    static std::map<json::string, classifier_model_type> map;
    static std::mutex mux;
    std::lock_guard lock(mux);
    const auto p = map.try_emplace(s,empty_rec);
    if (p.second)
        p.first->second.name = p.first->first;
    return &p.first->second;
}

static void
decode(face_coordinates_with_classifiers& fc, const json::value& top) {
    if (auto objp = decode(static_cast<face_coordinates&>(fc), top)) {
        auto& arrv = (*objp)[K_attr];
        if (json::is_type<json::array>(arrv)) {
            auto& arr = get_array(arrv);
            fc.classifiers.reserve(arr.size());
            for (auto& o : object_from_array(arr)) {
                auto p = empty_model(get_string(o[K_n]));
                std::vector<float> v;
                auto& vv = o[K_v];
                if (json::is_type<json::array>(vv)) {
                    auto& a = get_array(vv);
                    v.reserve(a.size());
                    for (auto& x : a)
                        v.push_back(make_number(x));
                }
                else if (vv != json::null)
                    v.push_back(make_number(vv));
                fc.classifiers.emplace_back(p,move(v));
            }
        }
    }
}

face_coordinates_with_classifiers::face_coordinates_with_classifiers(
    const json::value& v) {

    if (json::is_type<json::array>(v) || json::is_type<json::object>(v))
        decode(*this, v);
    else {
        auto bin = make_binary(v);
        if (bin.size() < 4)
            throw std::runtime_error(
                "invalid face_coordinates serialization (too small)");
        if (json::is_compressed(bin.data()))
            bin = json::pull_inflate_binary(bin).pull_final();
        decode(*this, json::decode_amf3_or_json(bin));
    }
}
