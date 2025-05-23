
#include "internal_cluster_1.hpp"
#include "internal_prototype_1.hpp"
#include "internal_serialize.hpp"
#include "model.hpp"

#include <core/thread_data.hpp>

#include <json/types.hpp>

#include <numeric>
#include <stdexcept>
#include <vector>

using namespace rec;
using namespace rec::internal;

static const prototype_1& p2_cast(const prototype_ptr& ptr) {
    if (!ptr) throw std::invalid_argument("prototype is nullptr");
    return *static_cast<const prototype_1*>(ptr.get());
}

static std::shared_ptr<const model_state>
get_model(const stdx::forward_iterator<prototype_ptr>& first,
          const stdx::forward_iterator<prototype_ptr>& last) {
    return first != last ? p2_cast(*first).model : nullptr;
}

static std::shared_ptr<const model_state>
get_model(const core::context_data& cd, const json::value& v) {
    stdx::binary bin;
    if (!json::is_type<json::array>(v))
        bin = make_binary(v);
    else {
        auto& arr = get_array(v);
        if (!arr.empty())
            bin = make_binary(arr.front());
    }
    if (bin.size() < 2)
        return {};
    auto* p = bin.data<uint8_t>();
    return core::get<context_map>(cd.context).get(p[0] ? p[0] : p[1]);
}

static std::vector<prototype_ptr>
construct_multiple(const core::context_data& cd, const json::value& v) {
    std::vector<prototype_ptr> result;
    if (json::is_type<json::array>(v)) {
        auto& arr = get_array(v);
        result.reserve(arr.size());
        for (auto& el : arr) {
            auto b = make_binary(el);
            result.emplace_back(
                prototype_1::deserialize(cd, b.data(), b.size()));
        }
    }
    else { // deprecated "serialize_multiple" binary format
        const auto src = make_binary(v);
        const auto bins =
            internal::deserialize_multiple(src.data(), src.size());
        result.reserve(bins.size());
        for (auto& b : bins)
            result.emplace_back(
                prototype_1::deserialize(cd, b.first, b.second));
    }
    return result;
}

static auto
compute_mean(const std::vector<prototype_ptr>& faces,
             const model_state& model) {
    std::vector<float> r;
    if (!faces.empty() &&
        std::fpclassify(model.l2sqr_max_score) == FP_NORMAL &&
        std::fpclassify(model.l2sqr_coeff) == FP_NORMAL &&
        model.l2sqr_max_score > 0 && model.l2sqr_coeff > 0)
        if (const auto n = p2_cast(faces.front()).get32_orig().second) {
            r.resize(n, 0);
            for (auto& fp : faces) {
                auto pr = p2_cast(fp).get32_orig();
                if (r.size() != pr.second)
                    throw std::logic_error("vector size mismatch");
                std::transform(r.begin(), r.end(),
                               pr.first, r.begin(),
                               [](float a, float b) { return a+b; });
            }
            for (auto& x : r)
                x /= float(faces.size());
        }
    return r;
}

static float
compute_boost(const std::vector<prototype_ptr>& faces) {
    std::vector<float> c;
    for (auto& f : faces) {
        auto v = p2_cast(f).get32_unit();
        if (c.empty()) {
            c.reserve(v.second);
            std::copy_n(v.first, v.second, back_inserter(c));
        }
        else {
            if (c.size() != v.second)
                throw std::logic_error("vector size mismatch");
            std::transform(c.begin(), c.end(),
                           v.first, c.begin(),
                           [](float a, float b) { return a+b; });
        }
    }
    if (c.empty()) return 1;
    auto n = std::sqrt(std::inner_product(
                           c.begin(), c.end(), c.begin(), 0.0f))
        / float(faces.size());
    if (n < 0.5) n = 0.5;
    return 1.0f / n;
}

cluster::cluster(const stdx::forward_iterator<prototype_ptr>& first,
                 const stdx::forward_iterator<prototype_ptr>& last)
    : model(get_model(first, last)),
      faces(first, last),
      mean_vec(compute_mean(faces,*model)),
      cos_boost(compute_boost(faces)) {
}

cluster::cluster(const core::context_data& cd, const json::value& v)
    : model(get_model(cd, v)),
      faces(construct_multiple(cd, v)),
      mean_vec(compute_mean(faces,*model)),
      cos_boost(compute_boost(faces)) {
}

float cluster::compare_to(const prototype& other, variant var) const {
    const auto& p = static_cast<const prototype_1&>(other);
    if (model.get() != p.model.get())
        throw std::logic_error("cannot compare prototypes from different context");
    if (comparison_class(var) == variant::none)
        var |= model->compare_variant.load(std::memory_order_relaxed);
    if (faces.empty())
        throw std::runtime_error("cluster corrupt (empty)");

    float s;
    switch (comparison_class(var)) {
    case variant::cos: {
        s = 0;
        float max = 0;
        for (auto& p : faces) {
            const auto r = compare(other, *p, var | variant::raw);
            if (max < r) max = r;
            s += r;
        }
        s /= float(faces.size());
        if (!(var & variant::raw)) {
            // ensure boosted average is not greater than max
            if ((s *= cos_boost) > max)
                s = max;
            s *= model->cos_max_score;
        }
        break;
    }

    case variant::l2sqr: {
        if (mean_vec.empty())
            throw std::runtime_error("prototype does not support L2 comparison");
        const auto pr = p.get32_orig();
        if (mean_vec.size() != pr.second)
            throw std::runtime_error("prototype corrupt (size mismatch)");
        s = -std::inner_product(
            mean_vec.begin(), mean_vec.end(), pr.first, 0.0f,
            [](auto a, auto b) { return a+b; },
            [](auto a, auto b) { return (a-b)*(a-b); });
        // note: s <= 0
        if (!(var & variant::raw))
            s = model->l2sqr_max_score + model->l2sqr_coeff * s;
        break;
    }

    default:
        throw std::invalid_argument("unsupported comparison class");
    }
    return s;
}

json::value cluster::diagnostic() const {
    static const auto K_class = json::string("class");
    static const auto K_ver = json::string("ver");
    static const auto K_size = json::string("size");
    static const auto K_boost = json::string("boost");
    json::object top;
    top[K_class] = "cluster_3";
    if (model)
        top[K_ver] = model->version;
    top[K_size] = size();
    top[K_boost] = cos_boost;
    return top;
}

json::array cluster::serialize() const {
    json::array arr;
    arr.reserve(faces.size());
    for (auto& face : faces)
        arr.emplace_back(
            to_binary(face,rec::raw,rec::uncompressed));
    return arr;
}
