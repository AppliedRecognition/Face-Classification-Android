
#include "types.hpp"

#include <json/types.hpp>
#include <json/zlib.hpp>

#include <stdext/rounding.hpp>

#include <applog/core.hpp>

#include <mutex>


using namespace det;


static const auto K_t = json::string("t");
static const auto K_c = json::string("c");
static const auto K_n = json::string("n");
static const auto K_v = json::string("v");
static const auto K_el = json::string("el");
static const auto K_er = json::string("er");
static const auto K_lm = json::string("lm");
static const auto K_lmf = json::string("lmf");
static const auto K_fcver = json::string("fcver");
static const auto K_attr = json::string("attr");
static const auto K_det = json::string("det");

// quarter pixel resolution is good enough
static json::value to_quarter(float v) {
    const auto i = stdx::round_to<long>(4*v);
    if (i&3) return double(i) / 4.0;
    return i/4;
}
static json::array to_array(const coordinate_type& p) {
    return { to_quarter(p.x), to_quarter(p.y) };
}

json::value det::to_json(const face_coordinates& fc) {
    json::array arr;
    arr.reserve(fc.size());
    for (auto& dc : fc) {
        json::array marks;
        marks.reserve(dc.landmarks.size());
        for (auto& mark : dc.landmarks)
            marks.push_back(to_array(mark));
        auto o = json::object {
            {K_t,  to_string(dc.type)},
            {K_c,  dc.confidence},
            {K_el, to_array(dc.eye_left)},
            {K_er, to_array(dc.eye_right)},
        };
        if (!marks.empty())
            o.insert({K_lm, move(marks)});
        arr.push_back(move(o));
    }
    return arr;
}

stdx::binary det::to_binary(const face_coordinates& fc, unsigned format) {
    const auto top = to_json(fc);

    stdx::binary r;
    if (format&2)
        r.assign(encode_json(top));
    else
        r = encode_amf3(top);

    if ((format&1) == 0)
        r = json::pull_deflate(r).pull_final();

    return r;
}

static coordinate_type coord_from_json(const json::value& v) {
    auto& a = get_array(v);
    if (a.size() < 2)
        throw std::runtime_error("invalid face_coordinates serialization (bad coordinate)");
    return { make_number(a[0]), make_number(a[1]) };
}

detected_coordinates::detected_coordinates(const json::value& v) {
    if (json::is_type<json::object>(v)) {
        // complete serialization
        auto& o = get_object(v);
        type = raw_image::dt_from_string(get_string_safe(o[K_t]));
        if (type == dt::unknown)
            FILE_LOG(logWARNING) << "unknown coordinate type: " << o[K_t];
        if (o[K_lmf] != json::null) {
            // flattened landmarks
            auto& a = get_array(o[K_lmf]);
            if (a.size() & 1)
                FILE_LOG(logWARNING) << "flattened landmarks have odd size: "
                                     << a.size();
            landmarks.resize(a.size()/2);
            auto src = a.begin();
            for (auto& p : landmarks) {
                p.x = make_number(*src++);
                p.y = make_number(*src++);
            }
            // assume eyes are not present
            set_eye_coordinates_from_landmarks();
            // if confidence not present then default to 10
            confidence = make_number(o[K_c], 10.0f);
        }
        else {
            confidence = make_number(o[K_c]);
            eye_left = coord_from_json(o[K_el]);
            eye_right = coord_from_json(o[K_er]);
            if (o[K_lm] != json::null) {
                auto& a = get_array(o[K_lm]);
                landmarks.reserve(a.size());
                for (auto& m : a)
                    landmarks.push_back(coord_from_json(m));
            }
        }
    }

    else if (json::is_type<json::array>(v)) {
        // serialization is of landmarks only
        auto& a = get_array(v);
        switch (a.size()) {
        case 68: type = dt::dlib68; break;
        case 7:  type = dt::v7_retina; break;
        case 5:  type = dt::dlib5;  break;
        default:
            throw std::runtime_error("invalid detected_coordinates serialization (incorrect number of landmarks)");
        }
        landmarks.reserve(a.size());
        for (auto& m : a)
            landmarks.push_back(coord_from_json(m));
        set_eye_coordinates_from_landmarks();
    }

    else
        throw std::runtime_error("invalid detected_coordinates serialization (not an array or object)");
}

static auto decode(face_coordinates& fc, const json::value& top) {
    json::object const* obj = nullptr;
    const auto& arr = [&]() -> const json::array& {
        if (json::is_type<json::array>(top))
            return get_array(top);
        obj = &get_object(top);
        if (get_integer_safe((*obj)[K_fcver]) != 1)
            throw std::runtime_error("invalid face_coordinates serialization (unknown version)");
        return get_array((*obj)[K_det]);
    }();
    fc.reserve(arr.size());
    for (auto& v : arr)
        fc.emplace_back(v);
    return obj;
}

face_coordinates::face_coordinates(const json::value& v) {
    if (json::is_type<json::array>(v) || json::is_type<json::object>(v))
        decode(*this, v);
    else {
        auto bin = make_binary(v);
        if (bin.size() < 4)
            throw std::runtime_error("invalid face_coordinates serialization (too small)");
        if (json::is_compressed(bin.data()))
            bin = json::pull_inflate_binary(bin).pull_final();
        decode(*this, json::decode_amf3_or_json(bin));
    }
}

face_coordinates::operator detected_coordinates&() & {
    if (empty())
        throw std::logic_error("face_coordinates is empty");
    return back();
}
face_coordinates::operator detected_coordinates&&() && {
    if (empty())
        throw std::logic_error("face_coordinates is empty");
    return std::move(back());
}
face_coordinates::operator const detected_coordinates&() const& {
    if (empty())
        throw std::logic_error("face_coordinates is empty");
    return back();
}
face_coordinates::operator const detected_coordinates&&() const&& {
    if (empty())
        throw std::logic_error("face_coordinates is empty");
    return std::move(back());
}
