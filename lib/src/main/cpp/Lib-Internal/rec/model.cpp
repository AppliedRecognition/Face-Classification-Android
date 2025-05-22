
#include "model.hpp"
#include "internal_prototype_1.hpp"

#include <applog/core.hpp>

using namespace rec;
using namespace rec::internal;


static constexpr auto proto1_static(
    version_type version, float cos_max_score,
    float l2sqr_max_score = 0, float l2sqr_coeff = 0) {
    return model_static {
        version, variant::cos, cos_max_score,
        l2sqr_max_score, l2sqr_coeff,
        &prototype_1::deserialize,
        &prototype_1::random
    };
}

static constexpr model_static known_models[] = {
    proto1_static(16, 5.5991041f, 5.75f, 8.0f),
    proto1_static(17, 5.6f),
    proto1_static(18, 5.6f),
    proto1_static(19, 5.6f),
    proto1_static(20, 5.925f),  // facenet-20170512
    proto1_static(21, 5.925f),
    proto1_static(22, 5.925f),
    proto1_static(23, 5.925f),
    proto1_static(24, 8.65f),   // arcnet (mobilefacenet)
};


/****************  class context_map  ****************/

context_map::context_map() {
    for (auto& m : known_models)
        map.emplace(m.version, std::make_shared<model_record>(m));
}

std::vector<version_type> context_map::known_versions() {
    std::vector<version_type> v;
    for (auto& m : known_models)
        v.push_back(m.version);
    return v;
}

std::shared_ptr<model_state> context_map::get(version_type ver) const {
    std::lock_guard<std::mutex> lock(mux);
    const auto it = map.find(ver);
    if (it == map.end() || !it->second) {
        FILE_LOG(logERROR) << "unknown recognition model version " << ver;
        throw std::invalid_argument("unknown recognition model version");
    }
    return it->second;
}

std::shared_ptr<const context_map::model_record>
context_map::find_or_load(version_type ver, typeinfo const* type,
                          std::function<std::shared_ptr<const void>()> loader) const {
    std::lock_guard<std::mutex> lock(mux);
    const auto it = map.find(ver);
    if (it == map.end() || !it->second)
        throw std::invalid_argument(
            "unknown recognition model version " + std::to_string(ver));
    auto& rec = *it->second;
    if (rec.m_type == nullptr) {
        rec.m_data = loader();
        if (!rec.m_data)
            throw std::runtime_error("recognition model loader failed");
        rec.m_type = type;
    }
    else if (rec.m_type != type)
        throw std::logic_error("recognition model has incorrect type");
    return it->second;
}

void context_map::insert(const model_static& model,
                         typeinfo const* type,
                         std::shared_ptr<const void> data) {
    const auto ver = model.version;
    if (ver <= 0) {
        FILE_LOG(logERROR) << "invalid custom recognition model";
        throw std::invalid_argument("invalid custom recognition model");
    }
    auto ptr = std::make_shared<model_record>(model);
    if (type && data) {
        ptr->m_data = move(data);
        ptr->m_type = type;
    }
    std::lock_guard<std::mutex> lock(mux);
    auto p = map.emplace(ver, move(ptr));
    if (!p.second) {
        FILE_LOG(logERROR) << "attempt to load custom recognition model with known version number " << ver;
        throw std::invalid_argument("recognition version number not available");
    }
}

