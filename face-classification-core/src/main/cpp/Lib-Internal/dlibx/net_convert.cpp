
#include "net_convert.hpp"
#include "net_layer_impl_inplace.hpp"
#include "net_layer_impl_tags.hpp"

#include <applog/core.hpp>

using namespace dlibx;


static auto get_params(const dlib::affine_& src) {
    // get_layer_params() is empty so we have to serialize / deserialize
    std::stringstream strm;
    serialize(src, strm);
    std::string version;
    dlib::deserialize(version, strm);
    if (version != "affine_" && version != "affine_2" )
        throw std::runtime_error(
            "unknown version (expected affine_ or affine_2)");
    dlib::resizable_tensor params;
    dlib::deserialize(params, strm);
    return params;
}

/// fuse affine that feeds into con
/// this only works with full-conv, not depth-wise
static bool
affine_into_con(const dlib::affine_& affine, net::layer_con& con) {
    const auto affine_params = get_params(affine);
    const auto in_channels = affine_params.size() / 2; // input filters
    if (in_channels <= 0) {
        FILE_LOG(logWARNING) << "remove_affine: affine layer is empty";
        return false;
    }
    if (affine_params.size() != 2*in_channels) {
        FILE_LOG(logWARNING) << "remove_affine: inconsistent affine parameters ("
                             << "params " << affine_params.size()
                             << " filters " << in_channels << ')';
        return false;
    }
    if (!con.add_bias()) {
        FILE_LOG(logWARNING) << "remove_affine: failed to add bias to conv layer";
        return false;
    }
    const auto num_filters = unsigned(con.num_filters());
    const auto image_size = unsigned(con.nr()*con.nc());
    const auto filter_size = in_channels*image_size;
    auto& con_params = con.get_layer_params();
    if (con_params.size() != (filter_size + 1) * num_filters) {
        FILE_LOG(logWARNING) << "remove_affine: inconsistent conv parameters ("
                             << "params " << con_params.size()
                             << " filters " << num_filters
                             << " size " << filter_size << ')';
        return false;
    }
    // y = bias + weight * (beta + gamma * x)
    // y = (bias + weight * beta) + (weight * gamma) * x
    auto gamma = affine_params.host();
    auto beta = gamma + in_channels;
    auto weight = con_params.host();
    auto bias = weight + filter_size*num_filters;
    for (auto nout = num_filters; nout > 0; --nout, ++bias) {
        // bias += beta * weight
        // weight *= gamma
        for (unsigned k = 0; k < in_channels; ++k)
            for (auto n = image_size; n > 0; --n, ++weight) {
                *bias += beta[k] * *weight;
                *weight *= gamma[k];
            }
    }
    return true;
}

/// fuse affine that follows con
static bool affine_into_con(net::layer_con& con, const dlib::affine_& affine) {
    const auto affine_params = get_params(affine);
    const auto num_filters = affine_params.size() / 2;
    if (num_filters <= 0) {
        FILE_LOG(logWARNING) << "remove_affine: affine layer is empty";
        return false;
    }
    if (affine_params.size() != 2*num_filters) {
        FILE_LOG(logWARNING) << "remove_affine: inconsistent affine parameters ("
                             << "params " << affine_params.size()
                             << " filters " << num_filters << ')';
        return false;
    }

    if (!con.add_bias()) {
        FILE_LOG(logWARNING) << "remove_affine: failed to add bias to conv layer";
        return false;
    }

    auto& con_params = con.get_layer_params();
    const auto filter_size = (con_params.size() / num_filters) - 1;
    if (con_params.size() != num_filters * (1+filter_size)) {
        FILE_LOG(logWARNING) << "remove_affine: inconsistent conv parameters ("
                             << "params " << con_params.size()
                             << " filters " << num_filters
                             << " size " << filter_size << ')';
        return false;
    }
    if (filter_size == 0) {
        FILE_LOG(logDETAIL) << "remove_affine: convolution has no filters (may be quantized)";
        return false;
    }

    // y = x * (conv * gamma) + (bias * gamma + beta)
    auto filt = con_params.host();
    auto bias = filt + filter_size*num_filters;
    auto gamma = affine_params.host();
    auto beta = gamma + num_filters;
    for (auto k = num_filters; k > 0; --k, ++bias, ++gamma, ++beta) {
        for (auto n = filter_size; n > 0; --n, ++filt)
            *filt *= *gamma;
        *bias = *bias * *gamma + *beta;
    }
    return true;
}

/// fuse affine that feeds into fc
static bool
affine_into_fc(const dlib::affine_& affine, net::layer_fc& fc) {
    const auto affine_params = get_params(affine);
    const auto in_channels = affine_params.size() / 2;
    if (in_channels <= 0) {
        FILE_LOG(logWARNING) << "remove_affine: affine layer is empty";
        return false;
    }

    if (!fc.add_bias()) {
        FILE_LOG(logWARNING) << "remove_affine: failed to add bias to fc layer";
        return false;
    }

    auto& fc_params = fc.get_layer_params();
    const auto out_channels = fc.get_num_outputs();
    const auto image_size =
        (fc_params.size() / out_channels - 1) / in_channels;
    if (affine_params.size() != 2*in_channels ||
        fc_params.size() != out_channels * (1+in_channels*image_size)) {
        FILE_LOG(logWARNING) << "remove_affine: inconsistent parameters ("
                             << "in " << in_channels
                             << "x" << image_size
                             << " out " << out_channels
                             << " affine " << affine_params.size()
                             << " fc " << fc_params.size() << ')';
        return false;
    }

    auto gamma = affine_params.host();
    auto beta = gamma + in_channels;
    auto weight = fc_params.host();  // in_channels are rows
    const auto bias = weight + in_channels*image_size*out_channels;
    const auto bend = bias + out_channels;

    // y = bias + weight * (beta + gamma * x)
    // y = (bias + weight * beta) + (weight * gamma) * x
    for (auto kin = in_channels; kin > 0; --kin, ++gamma, ++beta)
        for (auto n = image_size; n > 0; --n)
            for (auto b = bias; b != bend; ++b, ++weight) {
                *b += *weight * *beta;
                *weight *= *gamma;
            }
    assert(weight == bias);
    return true;
}

/// fuse affine that follows fc
static bool affine_into_fc(net::layer_fc& fc, const dlib::affine_& affine) {
    const auto affine_params = get_params(affine);
    const auto num_output = affine_params.size() / 2;
    if (num_output <= 0) {
        FILE_LOG(logWARNING) << "remove_affine: affine layer is empty";
        return false;
    }

    if (!fc.add_bias()) {
        FILE_LOG(logWARNING) << "remove_affine: failed to add bias to fc layer";
        return false;
    }

    auto& fc_params = fc.get_layer_params();
    const auto num_input = (fc_params.size() / num_output) - 1;
    if (affine_params.size() != 2*num_output ||
        fc_params.size() != num_output * (1+num_input)) {
        FILE_LOG(logWARNING) << "remove_affine: inconsistent parameters";
        return false;
    }

    // y = x * (fc * gamma) + (bias * gamma + beta)
    auto filt = fc_params.host();
    auto bias = filt + num_input*num_output;
    auto gamma = affine_params.host();
    auto beta = gamma + num_output;
    for (auto k = num_output; k > 0; --k, ++bias, ++gamma, ++beta, ++filt) {
        auto col = filt;
        for (auto n = num_input; n > 0; --n, col += num_output)
            *col *= *gamma;
        *bias = *bias * *gamma + *beta;
    }
    return true;
}

template <typename AP, typename CP>
static void update_downstream_inbound_names(
    const AP* to_remove, CP* new_inbound) {
    for (auto p : to_remove->outbound_nodes) {
        assert(p->inbound.size() == p->inbound_nodes.size());
        for (std::size_t i = 0; i < p->inbound.size(); ++i) {
            if (p->inbound_nodes[i] == to_remove) {
                assert(p->inbound[i] == to_remove->name);
                p->inbound_nodes[i] = new_inbound;
                p->inbound[i] = new_inbound->name;
            }
        }
    }
    // update new_inbound's outbound_nodes too
    auto& outbound = new_inbound->outbound_nodes;
    for (auto it = outbound.begin(), end = outbound.end(); it != end; ++it)
        if (*it == to_remove) {
            outbound.insert(
                outbound.erase(it),
                to_remove->outbound_nodes.begin(),
                to_remove->outbound_nodes.end());
            break;
        }
}

void net::remove_affine(std::vector<layer_ptr>& layers) {
    map_layers(layers.begin(), layers.end());
    for (auto it = layers.begin(); it != layers.end(); ) {
        if (const auto* ap = dynamic_cast<layer_affine*>(it->get())) {
            assert(ap->inbound_nodes.size() == 1);
            if (auto cp = dynamic_cast<layer_con*>(
                    ap->inbound_nodes.front())) {
                if (cp->outbound_nodes.size() == 1) {
                    assert(cp->outbound_nodes.front() == ap);
                    // can remove affine
                    if (affine_into_con(*cp, ap->detail)) {
                        update_downstream_inbound_names(ap,cp);
                        it = layers.erase(it);
                        continue; // don't do the ++it below
                    }
                }
            }
            else if (auto fcp = dynamic_cast<layer_fc*>(
                         ap->inbound_nodes.front())) {
                if (fcp->outbound_nodes.size() == 1) {
                    assert(fcp->outbound_nodes.front() == ap);
                    // can remove affine
                    if (affine_into_fc(*fcp, ap->detail)) {
                        update_downstream_inbound_names(ap,fcp);
                        it = layers.erase(it);
                        continue; // don't do the ++it below
                    }
                }
            }
            else if (ap->outbound_nodes.size() == 1) {
                if (auto cp = dynamic_cast<layer_con*>(
                        ap->outbound_nodes.front())) {
                    assert(cp->inbound_nodes.size() == 1 &&
                           cp->inbound_nodes.front() == ap);
                    // can remove affine that feeds into con
                    if (affine_into_con(ap->detail, *cp)) {
                        // update upstream outbound names
                        auto* p = ap->inbound_nodes.front();
                        cp->inbound = { p->name };
                        cp->inbound_nodes = { p };
                        for (auto& out : p->outbound_nodes)
                            if (out == ap)
                                out = cp;
                        it = layers.erase(it);
                        continue; // don't do the ++it below
                    }
                }
                else if (auto fcp = dynamic_cast<layer_fc*>(
                        ap->outbound_nodes.front())) {
                    assert(fcp->inbound_nodes.size() == 1 &&
                           fcp->inbound_nodes.front() == ap);
                    // can remove affine that feeds into fc
                    if (affine_into_fc(ap->detail, *fcp)) {
                        // update upstream outbound names
                        auto* p = ap->inbound_nodes.front();
                        fcp->inbound = { p->name };
                        fcp->inbound_nodes = { p };
                        for (auto& out : p->outbound_nodes)
                            if (out == ap)
                                out = fcp;
                        it = layers.erase(it);
                        continue; // don't do the ++it below
                    }
                }
            }
        }
        else if (const auto* mp = dynamic_cast<layer_multiply*>(it->get())) {
            assert(mp->inbound_nodes.size() == 1);
            if (auto cp = dynamic_cast<layer_con*>(
                    mp->inbound_nodes.front())) {
                if (cp->outbound_nodes.size() == 1) {
                    assert(cp->outbound_nodes.front() == mp);
                    // can remove multiply layer
                    cp->get_layer_params() *= mp->detail.get_multiply_value();
                    update_downstream_inbound_names(mp,cp);
                    it = layers.erase(it);
                    continue; // don't do the ++it below
                }
            }
            else if (auto fcp = dynamic_cast<layer_fc*>(
                         mp->inbound_nodes.front())) {
                if (fcp->outbound_nodes.size() == 1) {
                    assert(fcp->outbound_nodes.front() == mp);
                    // can remove multiply layer
                    fcp->get_layer_params() *= mp->detail.get_multiply_value();
                    update_downstream_inbound_names(mp,fcp);
                    it = layers.erase(it);
                    continue; // don't do the ++it below
                }
            }
        }
        ++it;
    }
}

void net::remove_dropout(std::vector<layer_ptr>& layers) {
    map_layers(layers.begin(), layers.end());
    for (auto it = layers.begin(); it != layers.end(); ) {
        auto& node = **it;
        if (node.code() == "invdropout") {
            assert(node.inbound_nodes.size() == 1);
            update_downstream_inbound_names(&node, node.inbound_nodes.front());
            it = layers.erase(it);
        }
        else
            ++it;
    }
}

unsigned net::serialize_native(
    stdx::span<const layer_ptr> layers, std::ostream& out) {

    unsigned count = 0;
    if (layers.empty()) return count;

    // check layers are mapped correctly
    if (!layers.front()->inbound_nodes.empty())
        throw std::logic_error("first layer must be an input layer");
    for (std::size_t idx = 1; idx < layers.size(); ++idx) {
        auto& l0 = *layers[idx-1];
        auto& l1 = *layers[idx];
        if (l0.outbound_nodes.empty() || l1.inbound_nodes.empty())
            throw std::logic_error("layers are not mapped");
        if (l0.outbound_nodes.front() != &l1 ||
            l1.inbound_nodes.front() != &l0)
            throw std::logic_error("layers are not mapped linearly");
    }

    using dlib::serialize;

    // note that the input layer is not wrapped in add_layer<>
    // so it is not included in the following loop
    for (auto idx = layers.size()-1; idx > 0; --idx) {
        if (layers[idx]->outbound_nodes.size() > 1) {
            serialize(1, out); // add_tag_layer
            ++count;
        }
        serialize(idx == 1 ? 3 : 2, out); // add_layer
        ++count;
    }

    // input layer
    ++count;
    if (layers.front()->outbound_nodes.size() <= 1)
        layers.front()->serialize_detail(out);

    else { // add_tag_layer<input>
        ++count;
        serialize(2, out);
        layers.front()->serialize_detail(out);
        serialize(empty_tensor, out); // cached_output
        serialize(empty_tensor, out); // grad_final
        serialize(true, out); // gradient_input_is_stale
        serialize(1u, out); // sample_expansion_factor
    }

    // computational layers
    for (std::size_t idx = 1; idx < layers.size(); ++idx) {
        auto& layer = *layers[idx];

        // add_prev, mult_prev and concat don't implement serialize_detail()
        const auto code = layer.code();
        if (code == "add_prev")
            serialize("add_prev_",out);
        else if (code == "mult_prev")
            serialize("mult_prev_",out);
        else if (code.compare(0,7,"concat_") == 0) {
            serialize("concat_",out);
            serialize(unsigned(atoi(code.c_str()+7)), out);
        }
        else // all other layers should serialize correctly
            layer.serialize_detail(out);

        serialize(true, out);  // this_layer_setup_called
        serialize(true, out);  // gradient_input_is_stale

        // get_output_and_gradient_input_disabled
        // if outbound_node is inplace, then true
        bool output_disabled = false;
        if (layer.outbound_nodes.size() == 1) {
            assert(idx + 1 < layers.size());
            auto& other = *layers[idx+1];
            if (dynamic_cast<const layer_inplace_base*>(&other))
                output_disabled = true;
        }
        serialize(output_disabled, out);

        serialize(empty_tensor, out); // x_grad
        serialize(empty_tensor, out); // cached_output
        serialize(empty_tensor, out); // params_grad
        if (idx == 1)
            serialize(1u, out); // sample_expansion_factor
    }

    return count;
}
