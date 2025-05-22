
#include "detection_settings.hpp"

#include <json/types.hpp>

#include <applog/core.hpp>

using namespace det;

json::value det::to_json(const detection_settings& settings) {
    json::array lma;
    if (lm::dlib5 & settings.landmark_detection.landmarks)
        lma.push_back("dlib5");
    if (lm::dlib68 & settings.landmark_detection.landmarks)
        lma.push_back("dlib68");
    if (lm::mesh68 & settings.landmark_detection.landmarks)
        lma.push_back("mesh68");
    if (lm::mesh478 & settings.landmark_detection.landmarks)
        lma.push_back("mesh478");
    json::object lmo = {
        { "landmarks", move(lma) },
        { "contrast_correction",
          settings.landmark_detection.contrast_correction },
    };
    json::object obj = {
        { "detector_version", settings.detector_version },
        { "size_range", settings.size_range },
        { "confidence_threshold", settings.confidence_threshold },
        { "landmark_detection", move(lmo) },
    };
    if (settings.detector_version)
        obj["limit_pose"] = settings.v3_limit_pose;
    if (settings.fast_scaling)
        obj["fast_scaling"] = settings.fast_scaling;
    return obj;
}

static void
from_json(detection_settings& ds, json::object const* ptr, bool require_all) {
    if (json::is_type<json::object>((*ptr)["detection"]))
        ptr = &get_object((*ptr)["detection"]);
    auto& obj = *ptr;

    const auto set1 = [&](auto& dest, auto src) {
        auto& v = obj[src];
        if (v != json::null)
            dest = make_number(v);
        else if (require_all) {
            FILE_LOG(logERROR) << "detection setting '" << src << "' not found";
            throw std::invalid_argument("detection setting not found");
        }
    };
    set1(ds.detector_version,     "detector_version");
    set1(ds.size_range,           "size_range");
    set1(ds.confidence_threshold, "confidence_threshold");

    const auto set2 = [&](auto& dest, auto src1, auto src2) {
        auto& v = obj[src1];
        if (v != json::null) {
            dest = make_number(v);
            return true;
        }
        else {
            auto& v = obj[src2];
            if (v != json::null) {
                dest = make_number(v);
                return true;
            }
            else if (require_all) {
                FILE_LOG(logERROR) << "detection setting '"
                                   << src1 << "' not found";
                throw std::invalid_argument("detection setting not found");
            }
        }
        return false;
    };
    if (ds.detector_version == 3 || !require_all) {
        auto& v = obj["limit_pose"];
        if (v != json::null)
            ds.v3_limit_pose = make_number(v);
        else {
            unsigned yaw_range_large = 1, yaw_range_small = 1;
            unsigned roll_range_large = 2, roll_range_small = 2;
            auto b = set2(yaw_range_large,  "yaw_range_large",  "yaw_range");
            b = set2(yaw_range_small,  "yaw_range_small",  "yaw_range") || b;
            b = set2(roll_range_large, "roll_range_large", "roll_range") || b;
            b = set2(roll_range_small, "roll_range_small", "roll_range") || b;
            if (b) {
                const auto y = yaw_range_small > 0 || yaw_range_large > 0;
                const auto r = roll_range_small > 1 || roll_range_large > 1;
                ds.v3_limit_pose = (y?0:1u) | (r?0:2u);
            }
        }
    }

    {   // don't throw if this option is not present
        auto& v = obj["fast_scaling"];
        if (v != json::null)
            ds.fast_scaling = make_number(v);
    }

    static const auto decode_lm = [](auto& arr) {
        auto r = lm::none;
        for (auto& v : arr) {
            auto& s = get_string(v);
            if (s == "dlib5")
                r = r + lm::dlib5;
            else if (s == "dlib68")
                r = r + lm::dlib68;
            else if (s == "mesh68")
                r = r + lm::mesh68;
            else if (s == "mesh478")
                r = r + lm::mesh478;
            else
                throw std::invalid_argument("unrecognized landmark detection option");
        }
        return r;
    };
    
    auto& lmv = obj["landmark_detection"];
    if (json::is_type<json::object>(lmv)) {
        auto& lm = get_object(lmv);
        auto& lma = lm["landmarks"];
        if (lma == json::null && require_all) {
            FILE_LOG(logERROR) << "detection setting 'landmarks' not found";
            throw std::invalid_argument("detection setting not found");
        }
        else {
            ds.landmark_detection.landmarks = decode_lm(get_array(lma));
            auto& cc = lm["contrast_correction"];
            if (cc == json::null && require_all) {
                FILE_LOG(logERROR) << "detection setting 'contrast_correction' not found";
                throw std::invalid_argument("detection setting not found");
            }
            else
                ds.landmark_detection.contrast_correction = make_number(cc);
        }
    }

    else if (json::is_type<json::array>(lmv))
        ds.landmark_detection.landmarks = decode_lm(get_array(lmv));

    else {
        auto& edv = obj["eye_detection_variant"];
        if (edv != json::null) {
            switch (get_integer(edv)&7) {
            case 0:
            case 3:
                ds.landmark_detection.landmarks = lm::none;
                break;
            case 1:
            case 2:
                ds.landmark_detection.landmarks = lm::dlib5;
                break;
            case 4:
            case 7:
                ds.landmark_detection.landmarks = lm::dlib68;
                break;
            case 5:
            case 6:
                ds.landmark_detection.landmarks = lm::dlib5 + lm::dlib68;
                break;
            }
        }
        else if (require_all) {
            FILE_LOG(logERROR) << "detection setting 'landmark_detection' not found";
            throw std::invalid_argument("detection setting not found");
        }
    }
}

void detection_settings::amend(const json::object& obj) {
    from_json(*this, &obj, false);
}

void detection_settings::assign(const json::object& obj) {
    from_json(*this, &obj, true);
}

detection_settings::detection_settings(const json::value& val) {
    from_json(*this, &get_object(val), true);
}
