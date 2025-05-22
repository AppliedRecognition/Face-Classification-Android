
#include "face_landmarks.hpp"
#include <map>

static constexpr auto s_v3_dlib = "v3_dlib";
static constexpr auto s_v4_dlib = "v4_dlib";
static constexpr auto s_v5_fapi = "v5_fapi";
static constexpr auto s_v6_rfb320 = "v6_rfb320";
static constexpr auto s_v7_retina = "v7_retina";
static constexpr auto s_v8_blaze = "v8_blaze";
static constexpr auto s_haar_eyes = "haar_eyes";
static constexpr auto s_stasm77 = "stasm77";
static constexpr auto s_dlib5 = "dlib5";
static constexpr auto s_dlib68 = "dlib68";
static constexpr auto s_mesh68 = "mesh68";
static constexpr auto s_mesh478 = "mesh478";
static constexpr auto s_unknown = "unknown";

using namespace raw_image;
using dt = detection_type;

std::string_view raw_image::to_string(dt t) {
    switch (t) {
    case dt::v3_dlib:   return s_v3_dlib;
    case dt::v4_dlib:   return s_v4_dlib;
    case dt::v5_fapi:   return s_v5_fapi;
    case dt::v6_rfb320: return s_v6_rfb320;
    case dt::v7_retina: return s_v7_retina;
    case dt::v8_blaze:  return s_v8_blaze;
    case dt::haar_eyes: return s_haar_eyes;
    case dt::stasm77:   return s_stasm77;
    case dt::dlib5:     return s_dlib5;
    case dt::dlib68:    return s_dlib68;
    case dt::mesh68:    return s_mesh68;
    case dt::mesh478:   return s_mesh478;
    default:            return s_unknown;
    }
}

static const std::map<std::string_view, dt> type_map = {
    { s_v3_dlib,   dt::v3_dlib },
    { s_v4_dlib,   dt::v4_dlib },
    { s_v5_fapi,   dt::v5_fapi },
    { s_v6_rfb320, dt::v6_rfb320 },
    { s_v7_retina, dt::v7_retina },
    { s_v8_blaze,  dt::v8_blaze },
    { s_haar_eyes, dt::haar_eyes },
    { s_stasm77,   dt::stasm77 },
    { s_dlib5,     dt::dlib5 },
    { s_dlib68,    dt::dlib68 },
    { s_mesh68,    dt::mesh68 },
    { s_mesh478,   dt::mesh478 },
};

dt raw_image::dt_from_string(std::string_view str) {
    const auto it = type_map.find(str);
    return it != type_map.end() ? it->second : dt::unknown;
}

float eye_coordinates::eye_distance() const {
    return std::sqrt(length_squared(eye_right - eye_left));
}

void landmark_coordinates::set_eye_coordinates_from_landmarks() {
    *static_cast<eye_coordinates*>(this) = eyes_subset(*this);
}
