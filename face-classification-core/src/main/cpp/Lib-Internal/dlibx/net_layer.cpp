
#include "net_layer_impl.hpp"

#include <applog/core.hpp>


using namespace dlibx;


/**************** class net::layer ****************/

dlib::tensor& net::layer::get_layer_params() {
    throw std::runtime_error("layer::get_layer_params() not implemented for this layer");
}
const dlib::tensor& net::layer::get_layer_params() const {
    return empty_tensor;
}

json::object net::layer::keras_object() const {
    return {};
}

json::array net::layer::keras_array() const {
    json::array a;
    auto o = keras_object();
    if (!o.empty()) a.push_back(move(o));
    return a;
}

void net::layer::to_tensor(stdx::forward_iterator<const raw_image::plane&>,
                           stdx::forward_iterator<const raw_image::plane&>) {
    throw std::logic_error(
        "input() not implemented (only available on input layers)");
}

dlib::tensor& net::layer::forward(
    stdx::forward_iterator<stdx::span<const raw_image::plane> > first,
    stdx::forward_iterator<stdx::span<const raw_image::plane> > last) {
    if (first == last)
        throw std::invalid_argument("forward called with no images");
    static const auto extract_front =
        [](auto&& span) -> auto const& {
            if (span.empty())
                throw std::invalid_argument("image has no planes");
            else if (span.size() != 1)
                throw std::runtime_error("mutli-frame input not implememnted");
            return span.front();
        };
    to_tensor({move(first),extract_front},{move(last),extract_front});
    if (!output_buffer)
        throw std::logic_error("to_tensor() did not produce output");
    return *(output_tensor = &*output_buffer);
}

void net::layer::forward_const(dlib::tensor const* const*, std::size_t) {
    throw std::logic_error(
        "forward() not implemented (not available on input layers)");
}

dlib::tensor& net::layer::forward_inplace(dlib::tensor& input) {
    dlib::tensor const* ptr = &input;
    forward_const(&ptr, 1);
    if (!output_buffer)
        throw std::logic_error("forward_const() did not produce output");
    return *output_buffer;
}

dlib::tensor& net::layer::forward(const dlib::tensor& input) {
    auto* p = &input;
    forward_const(&p, 1);
    if (!output_buffer)
        throw std::logic_error("forward_const() did not produce output");
    output_tensor = &*output_buffer;
    return *output_tensor;
}

dlib::tensor& net::layer::forward() {
    if (inbound_nodes.size() == 1 &&
        inbound_nodes.front()->outbound_nodes.size() == 1) {
        // can operate in place
        output_tensor = &forward_inplace(*inbound_nodes.front()->output_tensor);
    }
    else {
        input_tensors.clear();
        for (auto& ip : inbound_nodes)
            input_tensors.emplace_back(ip->output_tensor);
        forward_const(input_tensors.data(), input_tensors.size());
        if (!output_buffer)
            throw std::logic_error("forward_const() did not produce output");
        output_tensor = &*output_buffer;
    }
    return *output_tensor;
}

dlib::tensor& net::layer::assign_output(const dlib::tensor& input) {
    auto& t = allocate_output();
    t = input;
    return *(output_tensor = &t);
}

const dlib::tensor& net::layer::last_output() const {
    if (!output_tensor)
        throw std::logic_error("last_output() called before forward()");
    return *output_tensor;
}

static inline auto name_array3(json::value name) {
    return json::array{move(name),0,0};
}
static inline auto name_array4(json::value name) {
    return json::array{move(name),0,0,json::object{}};
}

json::object net::layer::keras(
    stdx::forward_iterator<const layer*> first,
    stdx::forward_iterator<const layer*> last) {

    json::object top;
    json::array layers;

    for ( ; first != last; ++first) {
        const auto arg = *first;
        if (!arg) throw std::invalid_argument("nullptr layer found");
        auto& layer = *arg;
        auto a = layer.keras_array();
        if (a.empty())
            a.push_back(json::object{});
        else
            for (std::size_t i = 1; i < a.size(); ++i) {
                auto& o0 = get_object(a[i-1]);
                auto& o1 = get_object(a[i]);
                auto& nv = o0["name"];
                if (!json::is_type<json::string>(nv))
                    nv = json::string{};
                auto& name = get_string(nv);
                if (!name.empty())
                    name = layer.name + '_' + name;
                else
                    name = layer.name + '_' + std::to_string(i-1);
                o1["inbound_nodes"] =
                    json::array{json::array{name_array4(nv)}};
            }
        get_object(a.back())["name"] = layer.name;
        json::array inbound;
        for (auto& name : layer.inbound)
            inbound.push_back(name_array4(name));
        if (!inbound.empty())
            inbound = json::array{move(inbound)};
        get_object(a.front())["inbound_nodes"] = move(inbound);
        for (auto& v : a)
            layers.push_back(move(v));
    }
    if (!layers.empty()) {
        top["input_layers"] =
            json::array{name_array3(get_object(layers.front())["name"])};
        top["output_layers"] =
            json::array{name_array3(get_object(layers.back())["name"])};
    }
    top["layers"] = move(layers);
    return top;
}

std::map<std::string_view, net::layer*> net::layer::map_layers(
    stdx::forward_iterator<layer*> first,
    stdx::forward_iterator<layer*> last) {

    std::map<std::string_view, layer*> map;

    for ( ; first != last; ++first) {
        auto lp = *first;
        if (!lp || lp->name.empty())
            throw std::invalid_argument("layer name is empty");
        auto& layer = *lp;
        layer.inbound_nodes.clear();
        layer.input_tensors.clear();
        layer.outbound_nodes.clear();
        layer.clear_output();
        if (!map.empty()) {
            if (layer.inbound.empty()) {
                FILE_LOG(logERROR) << "layer '" << lp->name
                                   << "' has no inputs";
                throw std::invalid_argument("layer has no input");
            }
            layer.inbound_nodes.reserve(layer.inbound.size());
            layer.input_tensors.reserve(layer.inbound.size());
            for (auto& name : layer.inbound) {
                auto it = map.find(name);
                if (it == map.end()) {
                    FILE_LOG(logERROR) << "cannot find input layer '"
                                       << name << "' for layer '"
                                       << lp->name << "'";
                    throw std::invalid_argument("layer input not found");
                }
                layer.inbound_nodes.emplace_back(it->second);
                it->second->outbound_nodes.emplace_back(&layer);
            }
        }
        else if (!layer.inbound.empty())
            throw std::invalid_argument("front (input) layer cannot accept input");
        if (!map.emplace(layer.name, &layer).second)
            throw std::invalid_argument("layer names are not unique");
    }

    return map;
}

net::layer_ptr net::layer::copy() const {
    auto r = copy_detail();
    r->name = name;
    r->inbound = inbound;
    return r;
}

net::layer const* net::layer::common_input(layer const* stop_at) const {
    if (inbound_nodes.size() <= 1)
        return inbound_nodes.empty() ? nullptr : inbound_nodes.front();
    // map node -> { branch count, next node }
    std::map<layer const*, std::pair<unsigned, layer const*> > map;
    // current position along each branch
    std::vector<layer const*> position(
        inbound_nodes.begin(), inbound_nodes.end());
    while (!position.empty()) {
        for (auto it = position.begin(); it != position.end(); ) {
            auto& p = *it;
            // advance to next node with fanout > 1
            // note: the node we're looking for has fanout > 1
            while (p && p != stop_at && p->outbound_nodes.size() <= 1)
                p = p->common_input(stop_at);
            if (!p)
                it = position.erase(it);
            else {
                // mark node as visited in map
                auto& r = map[p];
                if (++r.first >= inbound_nodes.size())
                    return p;
                if (p == stop_at)
                    it = position.erase(it);
                else {
                    // advance one node
                    if (!r.second)
                        r.second = p->common_input(stop_at);
                    p = r.second;
                    ++it;
                }
            }
        }
    }
    return nullptr; // shouldn't happen -- malformed graph
}

void net::layer::concise(std::ostream& out, layer const* stop_at) const {
    if (this == stop_at) {
        out << '@';
        return;
    }
    const auto input = common_input(stop_at);
    auto&& desc = layer_description();
    if (output_tensor) {
        if (input && input->output_tensor &&
            output_tensor->k() == input->output_tensor->k() &&
            output_tensor->nr() == input->output_tensor->nr() &&
            output_tensor->nc() == input->output_tensor->nc()) {
            // suppress output of size
        }
        else {
            out << output_tensor->k();
            if (output_tensor->nr() != 1 || output_tensor->nc() != 1)
                out << 'x' << output_tensor->nr() << 'x' << output_tensor->nc();
            out << '<';
        }
    }
    else if (desc.output_channels > 0)
        out << desc.output_channels << '<';
    out << desc.concise;
    if (inbound_nodes.size() == 1 && inbound_nodes.front() != stop_at) {
        out << '|';
        inbound_nodes.front()->concise(out, stop_at);
    }
    else if (inbound_nodes.size() > 1) {
        if (!input) {
            FILE_LOG(logWARNING) << "layer::concise() invalid tree structure";
            return;
        }
        auto sep = '(';
        for (auto p : inbound_nodes) {
            out << sep;
            sep = ',';
            p->concise(out, input);
        }
        out << ')';
        if (input != stop_at) {
            out << '|';
            input->concise(out, stop_at);
        }
    }
}

std::string net::layer::concise() const {
    std::stringstream ss;
    concise(ss);
    return ss.str();
}

parameter_format net::layer::parameter_format() const {
    return pf::native;  // default implementation
}

// note: net::layer::deserialize() is in net_layer_impl.cpp
