
#include "input_extractor_retina.hpp"

#include <applog/core.hpp>

#include <cassert>
#include <mutex>
#include <shared_mutex>
#include <map>
#include <unordered_map>

using namespace raw_image;

namespace {
    using unique_ptr = input_extractor::unique_ptr;
    using factory_method = input_extractor::factory_method;

    struct internal {
        std::map<std::string, factory_method, std::less<> > factory_map;
        std::unordered_map<std::string_view, unique_ptr> extractor_map;
        using shared_lock = std::shared_lock<std::shared_mutex>;
        using unique_lock = std::unique_lock<std::shared_mutex>;
        std::shared_mutex mux;

        internal() : factory_map {
                { "retina",    &retina_factory  },
                /*
                { "facechip",  &facechip_factory  },
                { "lm68chip",  &lm68chip_factory  },
                { "facedepth", &facedepth_factory },
                { "eyecrop",   &eyecrop_factory  },
                { "license",   &license_factory   },
                { "box",       &box_factory       },
                { "pointcloud", &pointcloud_factory }
                */
        } {}
    };
    internal* state = nullptr;

    struct input_extractor_init {
        input_extractor_init() {
            if (!state) state = new internal;
        }
        ~input_extractor_init() {
            delete state;
            state = nullptr;
        }
    };
    const input_extractor_init init;
}

void
input_extractor::register_factory(std::string prefix, factory_method factory) {
    if (!state) state = new internal;
    if (prefix.empty() || factory == nullptr)
        throw std::logic_error("invalid input_extractor prefix or factory");
    internal::unique_lock lock(state->mux);
    auto pr = state->factory_map.emplace(prefix, factory);
    if (!pr.second)
        throw std::logic_error(
            "attempt to register input_extractor factory more than once");
}

const input_extractor*
input_extractor::find(std::string_view name) {
    if (!state)
        return nullptr; // too early -- main() not called yet -- or too late
    factory_method fp = nullptr;
    if (auto lock = internal::shared_lock(state->mux)) {
        const auto it = state->extractor_map.find(name);
        if (it != state->extractor_map.end())
            return it->second.get();
        auto jt = state->factory_map.upper_bound(name);
        if (jt != state->factory_map.begin()) {
            --jt;
            if (name.compare(0, jt->first.size(), jt->first) == 0)
                fp = jt->second;
        }
    }
    if (fp) {
        internal::unique_lock lock(state->mux);
        // possible race: check again for entry
        const auto it = state->extractor_map.find(name);
        if (it != state->extractor_map.end())
            return it->second.get();
        // attempt to construct extractor using factory
        if (auto e = fp(name)) {
            auto p = state->extractor_map.emplace(e->name, move(e));
            assert(p.second);
            return p.first->second.get();
        }
    }
    FILE_LOG(logWARNING) << "unknown input extractor: " << name;
    return nullptr;
}

const input_extractor*
input_extractor::new_layout(raw_image::pixel_layout new_layout) const {
    if (new_layout == layout)
        return this; // no change necessary
    const auto from_layout =
        [rgba = (name.find("lm68") == 0)] (raw_image::pixel_layout layout)
        -> std::string_view {
            switch (layout) {
            case raw_image::pixel::gray8:  return "gray";
            case raw_image::pixel::a8:     return "depth";
            case raw_image::pixel::yuv:    return "yuv";
            case raw_image::pixel::rgb24:  return "rgb";
            case raw_image::pixel::rgba32: return rgba ? "rgba" : "rgbd";
            default: return {};
            }
        };
    const auto old_pixel = from_layout(layout);;
    const auto new_pixel = from_layout(new_layout);;
    if (old_pixel.empty() || new_pixel.empty())
        throw std::runtime_error(
            "failed to convert input extractor -- invalid pixel layout");
    const auto pos = name.find(old_pixel);
    if (pos <= 0 || name.size() <= pos)
        throw std::runtime_error(
            "failed to convert input extractor -- non-standard name");
    auto new_name = name.substr(0,pos);
    new_name += new_pixel;
    new_name += name.substr(pos + old_pixel.size()); // suffix
    return find(new_name);
}
