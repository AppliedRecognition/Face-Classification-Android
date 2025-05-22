
#include "internal_prototype_1.hpp"
#include "internal_multiface_2.hpp"
#include "internal_multiface_3.hpp"
#include "internal_serialize.hpp"
#include "model.hpp"

#include <core/thread_data.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "transcribe_16_to_20.ipp"


using namespace rec;
using namespace rec::internal;


/* Note regarding the integer std::inner_product calculations below.
 * Each vector element (in fpvc format) has a max absolute value of
 * 1708 (11-bits) so a product of 2 of them has a max abs value of
 * 2917264 (22-bits).
 * To fit within a 32-bit signed integer, the maximum number of these
 * that can be summed is 736.
 * Therefore, this is the max size of the vector.
 */

static_assert(sizeof(int) >= 4, "int too short");

static inline bool is_positive(float x) {
    return x > 0 && std::fpclassify(x) == FP_NORMAL;
}

static auto get_version(const std::shared_ptr<const model_state>& model) {
    if (!model) throw std::invalid_argument("model parameters are null");
    return model->version;
}

static uuid_type compute_uuid(
    const std::variant<internal::fpvc_vector_type,internal::fp16vec>& vec) {

    uuid_type result;
    assert((result.size() & 3) == 0);
    auto dest = reinterpret_cast<uint32_t*>(result.data());
    auto end = dest + result.size() / 4;
    if (auto p = std::get_if<internal::fpvc_vector_type>(&vec)) {
        if (p->second.empty())
            throw std::invalid_argument("prototype vector is empty");
        std::seed_seq seq(p->second.begin(), p->second.end());
        seq.generate(dest, end);
    }
    else if (auto p = std::get_if<internal::fp16vec>(&vec)) {
        std::seed_seq seq(p->begin(), p->end());
        seq.generate(dest, end);
    }
    else
        throw std::logic_error("variant must be one of its two types");
    return result;
}

std::shared_ptr<prototype_1> prototype_1::make_shared(
    std::shared_ptr<const model_state> model,
    std::variant<internal::fpvc_vector_type,internal::fp16vec> vec,
    const std::optional<uuid_type>& uuid) {
    std::size_t n = 0;
    if (auto p = std::get_if<internal::fpvc_vector_type>(&vec))
        n = p->second.size();
    else if (auto p = std::get_if<internal::fp16vec>(&vec))
        n = p->size();
    if (n == 128)
        return std::make_shared<prototype_1_final<128> >(move(model), move(vec), uuid);
    return std::make_shared<prototype_1_final<0> >(move(model), move(vec), uuid);
}

prototype_ptr prototype_1::deserialize(
    std::shared_ptr<const model_state> model, const void* src, std::size_t len,
    const std::optional<uuid_type>& uuid) {

    assert(model);
    auto vecs = internal::deserialize_fpvc(src, len);
    if (vecs.size() != 1)
        throw std::domain_error("invalid prototype serialization"
                                " (expected single vector)");
    auto& v = vecs.front();
    assert(!v.first.empty());
    if (!v.second.second.empty())
        return make_shared(move(model), move(v.second), uuid);
    else
        return make_shared(move(model), std::move(v.first), uuid);
}

prototype_ptr prototype_1::deserialize(
    const core::context_data& cd,
    const void* src, std::size_t len, const std::optional<uuid_type>& uuid) {
    if (len < 4)
        throw std::invalid_argument("prototype data too short");
    const auto ver = *static_cast<const unsigned char*>(src);
    return deserialize(
        core::get<context_map>(cd.context).get(ver), src, len, uuid);
}

namespace {
    struct seed_from_ptr {
        const void* ptr;
        inline operator unsigned() const {
            return unsigned(reinterpret_cast<std::size_t>(ptr)%4398046511093u);
        }
    };
}

prototype_ptr prototype_1::random(
    core::thread_data& td, std::shared_ptr<const model_state> model,
    const prototype* base, float score, variant var) {

    auto& rgen = core::get<std::mt19937>(td.thread, seed_from_ptr{&td.thread});
    std::normal_distribution<float> nd;

    if (!base) {
        std::array<float, 128> desc;  // !! todo: 128 assumed !!
        for (auto& x : desc)
            x = nd(rgen);
        auto v8 = fpvc_vector_compress(desc.begin(),desc.end());
        return make_shared(move(model), move(v8));
    }

    else if (var != variant::none)
        throw std::invalid_argument("variant not supported");

    else if (base->version != model->version)
        throw std::invalid_argument("prototype version mismatch");

    // else make a related prototype

    const auto orig = [&](){
        const auto itl = static_cast<const prototype_1*>(base)->get32_unit();
        std::vector<float> vec;
        vec.reserve(itl.second);
        std::copy_n(itl.first, itl.second, back_inserter(vec));
        return vec;
    }();

    const auto neg = score < 0;
    score = std::abs(score);
    const auto target_product = std::min(1.0f, score / model->cos_max_score);

    static const auto dot = [](const auto& a, const auto& b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    };

    // random perpendicular vector
    std::vector<float> desc;
    desc.reserve(orig.size());
    while (desc.size() < orig.size())
        desc.push_back(nd(rgen));
    {
        // element with largest absolute value
        const auto mel = std::size_t(
            std::max_element(
                orig.begin(), orig.end(),
                [](auto a, auto b) {
                    return std::abs(a) < std::abs(b);
                }) - orig.begin());
        desc[mel] -= dot(orig,desc) / orig[mel];
        const auto norm = std::sqrt(dot(desc,desc));
        assert(norm > 0);
        for (auto& x : desc)
            x /= norm;
    }

    // checks: orig and desc have norm 1 and are perpendicular
    assert(std::abs(dot(orig,orig) - 1) < 1e-4);
    assert(std::abs(dot(desc,desc) - 1) < 1e-4);
    assert(std::abs(dot(orig,desc)) < 1e-4);

    std::transform(desc.begin(), desc.end(), orig.begin(),
                   desc.begin(),
                   [ ca = std::sqrt(1-target_product*target_product),
                     cb = target_product](auto a, auto b) {
                       return ca*a + cb*b;
                   });

    // verify result
    assert(std::abs(dot(desc,desc) - 1) < 1e-4);
    assert(std::abs(dot(orig,desc) - target_product) < 1e-4);

    if (neg)
        for (auto& x : desc)
            x = -x;

    auto v8 = fpvc_vector_compress(desc.begin(),desc.end());
    return make_shared(move(model), move(v8));
}

std::unique_ptr<internal::multiface>
prototype_1::construct_multiface(float cluster_threshold) const {
    if (cluster_threshold <= 0)
        return std::make_unique<multiface_2>(version, cluster_threshold);
    else
        return std::make_unique<multiface_3>(version, cluster_threshold);
}

raw_image::plane_ptr
prototype_1::diagnostic_image(diagnostic diag, core::context_data*) const {
    switch (diag) {
    case diagnostic::extracted:
    case diagnostic::preprocessed:
        if (thumb) return std::make_unique<raw_image::plane>(*thumb);
        // fall through
    default:
        return nullptr;
    }
}

prototype_1::prototype_1(std::shared_ptr<const model_state> model,
                         const uuid_type& uuid)
    : prototype(get_version(model),uuid),
      model(move(model)) {
}


/****************  class prototype_1_final  ****************/

template <typename I>
static inline auto
inner_product_i16_n(const int16_t* a, const int16_t* b, I len) {
    int32_t sum = 0;
    for ( ; len > 0; --len, ++a, ++b)
        sum += *a**b;
    return sum;
}
static inline auto
inner_product_i16_128(const int16_t* a, const int16_t* b) {
    int32_t sum = 0;
    for (auto len = 128; len > 0; --len, ++a, ++b)
        sum += *a**b;
    return sum;
}

static inline auto norm(const internal::fp16vec& vec) {
    return float(std::sqrt(inner_product_i16_n(
                               vec.begin(), vec.begin(), vec.size())));
}

template <unsigned N>
static inline auto norm(const vec16_N<N>& v16) {
    return float(std::sqrt(inner_product_i16_n(
                               std::begin(v16.vec), std::begin(v16.vec),
                               std::size(v16.vec))));
}

namespace {
    struct move_vec8 {
        std::variant<internal::fpvc_vector_type,internal::fp16vec>& vec;

        inline operator internal::fpvc_vector_type() {
            if (auto p = std::get_if<internal::fpvc_vector_type>(&vec))
                return move(*p);
            return {};
        }

        template <std::size_t N>
        operator std::pair<float, std::array<unsigned char, N> >() const {
            if (auto p = std::get_if<internal::fpvc_vector_type>(&vec)) {
                if (p->second.size() != N)
                    throw std::invalid_argument("rec::prototype_1 incorrect vector size");
                std::pair<float, std::array<unsigned char, N> > r;
                r.first = p->first;
                std::copy(p->second.begin(), p->second.end(), r.second.begin());
                return r;
            }
            return {0,{}};
        }
    };
}

template <unsigned VEC_SIZE>
typename prototype_1_final<VEC_SIZE>::vec16_type
prototype_1_final<VEC_SIZE>::move_vec16(
    std::variant<internal::fpvc_vector_type,internal::fp16vec>& vec) const {

    vec16_type v;
    if (auto p = std::get_if<internal::fp16vec>(&vec)) {
        if (p->size() != VEC_SIZE)
            throw std::invalid_argument("invalid feature vector size");
        if (!is_positive(p->coeff))
            throw std::invalid_argument("invalid feature vector coefficient");
        v.coeff = p->coeff;
        std::copy(p->begin(), p->end(), v.vec);
    }
    else if (!is_positive(vec8.first))
        throw std::invalid_argument("invalid feature vector coefficient");
    else {
        v.coeff = vec8.first;
        std::copy(fpvc_s16_decompress_iterator(vec8.second.begin()),
                  fpvc_s16_decompress_iterator(vec8.second.end()),
                  v.vec);
    }
    return v;
}

template <>
typename prototype_1_final<0>::vec16_type
prototype_1_final<0>::move_vec16(
    std::variant<internal::fpvc_vector_type,internal::fp16vec>& vec) const {

    if (auto p = std::get_if<internal::fp16vec>(&vec)) {
        if (!(16 <= p->size() && p->size() <= 512))
            throw std::invalid_argument("invalid feature vector size");
        if (!is_positive(p->coeff))
            throw std::invalid_argument("invalid feature vector coefficient");
        return std::move(*p);
    }
    if (!(0 < vec8.second.size() && vec8.second.size() <= 512))
        throw std::invalid_argument("invalid feature vector size");
    if (!is_positive(vec8.first))
        throw std::invalid_argument("invalid feature vector coefficient");
    internal::fp16vec v;
    v.coeff = vec8.first;
    v.resize(vec8.second.size());
    std::copy(fpvc_s16_decompress_iterator(vec8.second.begin()),
              fpvc_s16_decompress_iterator(vec8.second.end()),
              v.begin());
    return v;
}

template <unsigned VEC_SIZE>
prototype_1_final<VEC_SIZE>::prototype_1_final(
    std::shared_ptr<const model_state> model,
    std::variant<internal::fpvc_vector_type,internal::fp16vec> vec,
    const std::optional<uuid_type>& uuid)
    : prototype_1(model, uuid ? *uuid : compute_uuid(vec)),
      vec8(move_vec8{vec}),
      vec16(move_vec16(vec)),
      invnorm16(1 / norm(vec16)) {
}

template <unsigned VEC_SIZE>
prototype_1_final<VEC_SIZE>::prototype_1_final(
    std::shared_ptr<const model_state> model,
    const vec8_type& vec8, const vec16_type& vec16, const uuid_type& uuid)
    : prototype_1(model, uuid),
      vec8(vec8),
      vec16(vec16),
      invnorm16(1 / norm(vec16)) {
}

template <unsigned VEC_SIZE>
prototype_ptr prototype_1_final<VEC_SIZE>::copy(const std::optional<uuid_type>& new_uuid) const {
    return std::make_shared<prototype_1_final>(
        model, vec8, vec16, new_uuid ? *new_uuid : uuid);
}

template <unsigned VEC_SIZE>
inline std::tuple<stdx::span<const int16_t>,float,float>
prototype_1_final<VEC_SIZE>::get16() const {
    return { { vec16.vec, VEC_SIZE }, vec16.coeff, invnorm16 };
}

template <>
inline std::tuple<stdx::span<const int16_t>,float,float>
prototype_1_final<0>::get16() const {
    return { { vec16.begin(), vec16.size() }, vec16.coeff, invnorm16 };
}

/* Formats:
 *   0 (default) FPVC if in prototype, otherwise no-loss bit rate
 *   1 -> short 8-bit x 128 format if VEC_SIZE = 128
 *        otherwise 12-bit
 *   2 -> 12-bit
 *   3 -> 16-bit
 */
template <unsigned VEC_SIZE>
stdx::binary prototype_1_final<VEC_SIZE>::serialize() const {
    const auto f = model->serialize_format.load();
    if (f < 1 && 0 < vec8.first) {
        fpvc_vector_type v;
        v.first = vec8.first;
        v.second.assign(vec8.second.begin(), vec8.second.end());
        return internal::serialize(version, &v, (&v)+1);
    }
    fp16vec v;
    v.coeff = vec16.coeff;
    v.resize(VEC_SIZE);
    std::copy_n(vec16.vec, VEC_SIZE, v.begin());
    if (VEC_SIZE == 128 &&
        (f == 1 || (f == 0 && bits_required(v) <= 8))) {
        serialize_buffer_type buf;
        buf.reserve(132);
        serialize_value<uint16_t>(buf, uint16_t(version + 256));
        serialize_8(buf, v);
        assert(buf.size() == 132);
        return buf;
    }
    return internal::serialize(version, &v, (&v)+1, f < 3 ? 12 : 16);
}
template <>
stdx::binary prototype_1_final<0>::serialize() const {
    const auto f = model->serialize_format.load();
    if (f >= 3)
        return internal::serialize(version, &vec16, (&vec16)+1, 16);
    else if (f >= 1 || vec8.second.empty())
        return internal::serialize(version, &vec16, (&vec16)+1, 12);
    else
        return internal::serialize(version, &vec8, (&vec8)+1);
}

template <unsigned VEC_SIZE>
prototype_ptr prototype_1_final<VEC_SIZE>::transcribe_to(
    const core::context_data& cd, version_type target) const {
    const auto v16 = std::get<0>(get16());
    if (version == 16 && target == 20 && v16.size() == 128) {
        std::vector<float> desc(v16.size());
        transcribe_16_to_20(v16.begin(), invnorm16, desc.data());
        auto v8 = internal::fpvc_vector_compress(desc.begin(),desc.end());
        return std::make_shared<prototype_1_final>(
            core::get<context_map>(cd.context).get(target), move(v8));
    }
    throw std::runtime_error("transcribe not available for this version");
}

template <unsigned VEC_SIZE>
std::pair<stdx::forward_iterator<float>,unsigned>
prototype_1_final<VEC_SIZE>::vector_for_pca(unsigned i) const {
    if (i > 0)
        throw std::runtime_error("prototype corrupt (too short)");
    const auto v16 = std::get<0>(get16());
    return std::pair<stdx::forward_iterator<float>,unsigned> {
        { v16.begin(), [c=invnorm16](auto x) { return c*x; } },
        unsigned(v16.size())
    };
}

template <unsigned VEC_SIZE>
compare_result
prototype_1_final<VEC_SIZE>::compare_to(const prototype& other, variant var) const {
    const auto& p = static_cast<const prototype_1&>(other);
    if (model.get() != p.model.get())
        throw std::logic_error("cannot compare prototypes from different context");
    if (comparison_class(var) == variant::none)
        var |= model->compare_variant.load(std::memory_order_relaxed);

    float s;
    switch (comparison_class(var)) {
    case variant::cos: {
        if (!is_positive(model->cos_max_score))
            throw std::runtime_error("prototype does not support cosine comparison");
        const auto v16 = std::get<0>(get16());
        const auto ot = p.get16();
        if (v16.size() != std::get<0>(ot).size())
            throw std::runtime_error("prototype corrupt (size mismatch)");
        int32_t x;
        if (VEC_SIZE == 128 || v16.size() == 128)
            x = inner_product_i16_128(v16.begin(), std::get<0>(ot).begin());
        else // generic version
            x = inner_product_i16_n(v16.begin(), std::get<0>(ot).begin(),
                                    v16.size());
        s = float(x) * invnorm16 * std::get<2>(ot);
        if (!(var & variant::raw))
            s *= model->cos_max_score;
        break;
    }

    case variant::l2sqr: {
        if (!(var & variant::raw) &&
            !(is_positive(model->l2sqr_max_score) &&
              is_positive(model->l2sqr_coeff)))
            throw std::runtime_error("prototype does not support L2 comparison");
        const auto a = get32_orig();
        const auto b = p.get32_orig();
        if (a.second != b.second)
            throw std::runtime_error("prototype corrupt (size mismatch)");
        auto aend = a.first;
        advance(aend, a.second);
        s = -std::inner_product(
            a.first, aend, b.first, 0.0f,
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
