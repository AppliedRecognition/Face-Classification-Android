
#include "net_vector.hpp"
#include "net_layer.hpp"
#include "hash32.hpp"

#include "input_extractor_facechip.hpp"
#include "input_extractor_license.hpp"
#include "input_extractor_eyecrop.hpp"
#include "input_extractor_box.hpp"

#include <applog/core.hpp>


using namespace dlibx;
using net::layer_ptr_vector;


/**************** class net::vector ****************/

net::vector::vector() = default;
net::vector::~vector() = default;

net::vector::vector(layer_ptr_vector&& layers) : m_layers(move(layers)) {
    map_layers(m_layers.begin(), m_layers.end());
}

net::vector::vector(vector&&) = default;
net::vector& net::vector::operator=(vector&&) = default;

net::vector::vector(const vector& other)
    : meta(other.meta),
      labels(other.labels),
      input_extractor(other.input_extractor) {
    m_layers.reserve(other.size());
    for (auto& layer : other)
        m_layers.emplace_back(layer.copy());
    map_layers(m_layers.begin(), m_layers.end());
}

net::vector::vector(std::istream& in) {
    deserialize(in);
}

net::vector& net::vector::operator=(const vector& other) {
    if (this != &other) // copy construct and move
        operator=(net::vector(other));
    return *this;
}

void net::vector::set_layers(layer_ptr_vector&& new_layers) {
    map_layers(new_layers.begin(), new_layers.end());
    m_layers = move(new_layers);
}

layer_ptr_vector net::vector::release_layers() {
    return move(m_layers);
}
net::vector::operator layer_ptr_vector() && {
    return move(m_layers);
}

raw_image::plane_ptr
net::vector::extract(const multi_plane_arg& image,
                     const std::vector<point2f>& pts) const {
    if (image.empty())
        throw std::invalid_argument("empty image passed to extract");
    if (pts.empty())
        throw std::invalid_argument("empty landmarks passed to extract");
    if (!input_extractor)
        throw std::logic_error("input_extractor is null in extract");
    if (auto r = (*input_extractor)(image,pts))
        return r;
    throw std::runtime_error("image extraction failed");
}

static auto output_diagnostic(const dlib::tensor& t, const net::layer& layer) {
    std::string s;
    s.reserve(48);
    s += hash32({
            reinterpret_cast<const char*>(t.host()),
            std::size_t(4*t.k()*t.nr()*t.nc())
        });
    s += ':';
    s += std::to_string(t.k());
    s += 'x';
    s += std::to_string(t.nr());
    s += 'x';
    s += std::to_string(t.nc());
    s += ':';
    s += layer.code();
    return s;
}

std::pair<float const*, std::size_t>
net::vector::apply(const multi_frame_span& img, json::array* diag) {
    if (empty())
        throw std::logic_error("net::vector is empty");
    if (diag) {
        diag->clear();
        diag->reserve(size());
    }
    auto it = m_layers.begin();
    dlib::tensor const* out = &(*it)->forward(&img, &img+1);
    if (diag) diag->emplace_back(output_diagnostic(*out,**it));
    for (auto end = m_layers.end(); ++it != end; ) {
        out = &(*it)->forward();
        if (diag) diag->emplace_back(output_diagnostic(*out,**it));
    }
    assert(out && out->num_samples() == 1);
    return { out->host(), out->size() };
}
std::size_t
net::vector::operator()(const multi_frame_span& img,
                        stdx::span<dlib::resizable_tensor> dest) {
    std::size_t n = 0;
    if (!empty()) {
        auto it = m_layers.begin();
        dlib::tensor const* out = &(*it)->forward(&img, &img+1);
        for (auto end = m_layers.end(); ++it != end; ) {
            auto& layer = **it;
            out = &layer.forward();
            if (layer.outbound_nodes.empty()) {
                dest[n] = *out;
                if (++n >= dest.size())
                    break;
            }
        }
    }
    return n;
}
std::size_t
net::vector::operator()(const dlib::tensor& input,
                        stdx::span<dlib::resizable_tensor> dest) {
    std::size_t n = 0;
    if (!empty()) {
        auto it = m_layers.begin();
        dlib::tensor const* out = &(*it)->assign_output(input);
        for (auto end = m_layers.end(); ++it != end; ) {
            auto& layer = **it;
            out = &layer.forward();
            if (layer.outbound_nodes.empty()) {
                dest[n] = *out;
                if (++n >= dest.size())
                    break;
            }
        }
    }
    return n;
}

std::pair<float const*, std::size_t>
net::vector::apply(
    stdx::forward_iterator<multi_frame_span>& first,
    const stdx::forward_iterator<multi_frame_span>& last) {
    if (empty())
        throw std::logic_error("net::vector is empty");
    auto it = m_layers.begin();
    dlib::tensor const* out = &(*it)->forward(first, last);
    for (auto end = m_layers.end(); ++it != end; )
        out = &(*it)->forward();
    assert(out);
    return { out->host(), out->k() * out->nr() * out->nc() };
}
void net::vector::operator()(stdx::forward_iterator<multi_frame_span> first,
                             stdx::forward_iterator<multi_frame_span> last,
                             dlib::resizable_tensor& dest) {
    if (empty())
        throw std::logic_error("net::vector is empty");
    const auto n = distance(first, last);
    if (n > 0) {
        auto it = m_layers.begin();
        dlib::tensor const* out = &(*it)->forward(first, last);
        for (auto end = m_layers.end(); ++it != end; )
            out = &(*it)->forward();
        assert(out && out->num_samples() == n);
        dest = *out;
    }
    else
        dest.clear();
}

std::pair<std::string, unsigned long>
net::vector::output_type_and_size() const {
    std::pair<std::string, unsigned long> r{{},0};
    if (!empty()) {
        auto* layer = m_layers.back().get();
        while (r.first.empty() || r.second == 0) {
            const auto p = layer->layer_type_and_output_size();
            if (r.first.empty() && !p.first.empty())
                r.first = p.first;
            if (r.second == 0)
                r.second = p.second;
            if (layer->inbound_nodes.size() == 1)
                layer = layer->inbound_nodes.front();
            else break;
        }
    }
    return r;
}

json::object net::vector::description() const {
    json::object top;

    if (auto e = input_extractor)
        top["input"] = json::object {
            { "name",   e->name },
            { "width",  e->width },
            { "height", e->height },
            { "pixel",  to_string(e->layout) }
        };
    else
        top["input"] = json::null;

    top["labels"] = labels;
    top["layers"] = size();

    {
        auto ts = output_type_and_size();
        json::object output;
        if (!ts.first.empty())
            output["type"] = ts.first;
        if (ts.second)
            output["size"] = ts.second;
        top["output"] = move(output);
    }

    json::object m;
    for (auto& p : meta)
        if (p.first == "meta" || !top.try_emplace(p.first, p.second).second)
            m[p.first] = p.second;
    if (!m.empty())
        top["meta"] = move(m);

    return top;
}

std::string net::vector::concise() const {
    return (!empty() && m_layers.back())
        ? m_layers.back()->concise() : std::string{};
}

void net::vector::serialize(std::ostream& out) const {
    const auto version = -65;  // version 'A' encodes as 0x81 0x41
    dlib::serialize(version, out);

    std::string meta_enc;
    encode_json(meta_enc, meta);
    dlib::serialize(meta_enc, out);

    dlib::serialize(labels, out);

    auto extractor = input_extractor ? input_extractor->name : "";
    dlib::serialize(extractor, out);

    dlib::serialize(m_layers, out);
}

void net::vector::deserialize(std::istream& in) {
    int version = 0;
    dlib::deserialize(version, in);
    if (version != -65)
        throw dlib::serialization_error("unknown net::vector version");

    std::string meta_enc;
    dlib::deserialize(meta_enc, in);
    auto meta_value = json::decode_json(meta_enc);
    auto& new_meta = get_object(meta_value);

    std::vector<std::string> new_labels;
    dlib::deserialize(new_labels, in);

    std::string extractor;
    dlib::deserialize(extractor, in);
    auto new_extractor = !extractor.empty() ?
        raw_image::input_extractor::find(extractor) : nullptr;
    
    layer_ptr_vector new_layers;
    dlib::deserialize(new_layers, in);
    map_layers(new_layers.begin(), new_layers.end());

    meta = move(new_meta);
    labels = move(new_labels);
    input_extractor = new_extractor;
    m_layers = move(new_layers);
}


/****************  register input_extractor  ****************/

namespace {
    struct input_extractor_init {
        input_extractor_init() {
            const auto reg = [](auto prefix, auto factory) {
                raw_image::input_extractor::register_factory(
                    std::move(prefix), factory);
            };
            reg("facechip",  &facechip_factory);
            reg("lm68chip",  &lm68chip_factory);
            reg("facedepth", &facedepth_factory);
            reg("eyecrop",   &eyecrop_factory);
            reg("license",   &license_factory);
            reg("box",       &box_factory);
        }
    };
    const input_extractor_init init;
}
