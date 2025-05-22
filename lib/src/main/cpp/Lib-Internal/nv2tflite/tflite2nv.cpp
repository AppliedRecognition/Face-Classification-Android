
#include "conv_tools.hpp"
#include "conv_tflite.hpp"
#include "tflite_infer.hpp"

#include <dlibx/net_vector.hpp>
#include <dlibx/net_layer.hpp>
#include <dlibx/net_layer_impl.hpp>
#include <dlibx/input_extractor.hpp>
#include <dlibx/library_init.hpp>

#include <raw_image/io.hpp>
#include <raw_image/transform.hpp>
#include <raw_image/face_landmarks.hpp>

#include <json/types.hpp>

#include <applog/core.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>


using namespace conv;

template <typename VEC>
static unsigned remove_output(VEC& layers, dlibx::net::layer* node) {
    if (!node || !node->outbound_nodes.empty())
        return 0;
    auto find = [&](auto* ptr) {
        for (auto it = layers.begin(), end = layers.end(); it != end; ++it)
            if (it->get() == ptr)
                return it;
        throw std::runtime_error("not not found in layers vector");
    };
    const auto jt = find(node);
    const auto inbound = node->inbound_nodes;
    for (auto* ptr : inbound) {
        const auto end =
            remove(ptr->outbound_nodes.begin(),
                   ptr->outbound_nodes.end(), node);
        assert(distance(end, ptr->outbound_nodes.end()) == 1);
        ptr->outbound_nodes.erase(end, ptr->outbound_nodes.end());
    }
    layers.erase(jt);
    unsigned count = 1;
    for (auto* ptr : inbound)
        count += remove_output(layers, ptr);
    return count;
}

template <typename CONV>
static unsigned
output_subset(CONV& detail_orig,
              unsigned first, unsigned incr = 1,
              unsigned last = unsigned(-1)) {
    assert(0 < incr);
    const auto k_orig = unsigned(detail_orig.num_filters());
    assert(first < k_orig);
    assert(k_orig % incr == 0);
    if (k_orig < last)
        last = k_orig + first % incr;
    const auto k_new = k_orig - (last - first) / incr;
    assert(0 < k_new && k_new < k_orig);

    const auto filter_size = unsigned(detail_orig.nr()*detail_orig.nc());
    auto const& params_orig = detail_orig.get_layer_params();
    const auto num_inputs = (params_orig.size()/k_orig - 1) / filter_size;
    assert(params_orig.size() == (num_inputs*filter_size+1) * k_orig);

    CONV detail_new(k_new);
    struct sub {
        dlib::resizable_tensor t;
        auto& get_output() const { return t; }
    };
    detail_new.setup(sub{dlib::resizable_tensor{
                1, long(num_inputs), detail_orig.nr(), detail_orig.nc()
            }});
    auto& params_new = detail_new.get_layer_params();
    assert(params_new.size() == (num_inputs*filter_size+1) * k_new);

    // weights
    float const* src = params_orig.host();
    float* dest = params_new.host_write_only();
    for (unsigned k = 0; k < k_orig; ++k, src += num_inputs*filter_size)
        if (k < first || last <= k || (k - first) % incr != 0)
            dest = std::copy_n(src, num_inputs*filter_size, dest);

    // biases
    for (unsigned k = 0; k < k_orig; ++k, ++src)
        if (k < first || last <= k || (k - first) % incr != 0)
            *dest++ = *src;
    assert(src == params_orig.host() + params_orig.size());
    assert(dest == params_new.host() + params_new.size());

    detail_orig = detail_new;
    return 0;
}

template <typename CONV>
static unsigned
output_subset(CONV& detail_orig, stdx::span<const unsigned> indices) {
    const auto k_orig = unsigned(detail_orig.num_filters());
    for (auto k : indices)
        assert(k < k_orig);
    const auto k_new = unsigned(indices.size());
    assert(0 < k_new && k_new < k_orig);
    
    const auto filter_size = unsigned(detail_orig.nr()*detail_orig.nc());
    auto const& params_orig = detail_orig.get_layer_params();
    const auto num_inputs = (params_orig.size()/k_orig - 1) / filter_size;
    assert(params_orig.size() == (num_inputs*filter_size+1) * k_orig);

    CONV detail_new(k_new);
    struct sub {
        dlib::resizable_tensor t;
        auto& get_output() const { return t; }
    };
    detail_new.setup(sub{dlib::resizable_tensor{
                1, long(num_inputs), detail_orig.nr(), detail_orig.nc()
            }});
    auto& params_new = detail_new.get_layer_params();
    assert(params_new.size() == (num_inputs*filter_size+1) * k_new);

    // weights
    float const* src = params_orig.host();
    float* dest = params_new.host_write_only();
    for (auto k : indices)
        dest = std::copy_n(
            src + k*num_inputs*filter_size, num_inputs*filter_size, dest);
    // biases
    src += k_orig*num_inputs*filter_size;
    for (auto k : indices)
        *dest++ = src[k];
    assert(dest == params_new.host() + params_new.size());

    detail_orig = detail_new;
    return 0;
}


// **** main ****

int main(int argc, char*argv[]) {
    const char* const prog = [&](){
        assert(argc > 0);
        if (auto p = strrchr(argv[0], '/')) return p + 1;
        return argv[0];
    }();
    ++argv, --argc;

    if (argc <= 0) {
        FILE_LOG(logFATAL) << "Usage:" << std::endl
                           << '\t' << prog << " model_file.tflite";
        return 1;
    }
    const auto tflite_path = std::filesystem::path(*argv);
    ++argv, --argc;
    if (!is_regular_file(tflite_path)) {
        FILE_LOG(logFATAL) << "file not found: " << tflite_path;
        return 1;
    }

    const auto nv_path = tflite_path.filename().replace_extension(".nv");
    if (exists(nv_path)) {
        FILE_LOG(logERROR) << "destination path exists:" << nv_path;
        return 1;
    }

    // library init
    dlibx::library_init();

    auto tflite_model = conv::tflite_model(tflite_path);

    tflite_model.log_metadata(logINFO);
    
    auto const& opcodes = tflite_model.opcodes;
    const auto& sg_tensors = tflite_model.sg_tensors;
    const auto& sg_operators = tflite_model.sg_operators;

    assert(tflite_model.input_shape[1] == tflite_model.input_shape[2]);
    const auto input_dim = tflite_model.input_shape[1];

    auto& tensors = tflite_model.tensors;

    // copy float32 and int32 parameter tensors
    FILE_LOG(logINFO) << "copy float32 and int32 params";
    tflite_model.copy_float32_and_int32_params();

    // dequantize parameter tensors
    FILE_LOG(logINFO) << "dequantize params";
    tflite_model.dequantize_params();


    const auto prefix = std::string("l");
    std::set<std::string> layer_names;
    std::vector<dlibx::net::layer_ptr> dest_layers;

    {
        // input layer
        using image_type = dlib::matrix<dlib::rgb_pixel>;
        using input_type = dlibx::input_generic_image<image_type>;
        dest_layers.emplace_back(
            std::make_unique<dlibx::net::layer_input<input_type> >(
                dlibx::input_normalization::zero_center));
        dest_layers.back()->name = prefix + "0";
    }

    // process computation layers
    FILE_LOG(logINFO) << "process computation layers";
    for (auto* ptr : sg_operators) {
        tflite::Operator const& op = *ptr;

        assert(op.opcode_index() < opcodes.size());
        auto* opcode_ptr = opcodes[op.opcode_index()];
        assert(opcode_ptr);
        tflite::OperatorCode const& opcode = *opcode_ptr;
        if (builtin_code(opcode) == tflite::BuiltinOperator_DEQUANTIZE)
            continue;

        auto* out_ptr = op.outputs();
        assert(out_ptr && out_ptr->size() == 1);
        const auto out_idx = (*out_ptr)[0];
        assert(0 <= out_idx && unsigned(out_idx) < sg_tensors.size());
        auto* dest_ptr = sg_tensors[unsigned(out_idx)];
        assert(dest_ptr);
        tflite::Tensor const& dest = *dest_ptr;
        auto* dest_shape_ptr = dest.shape();
        assert(dest_shape_ptr);
        const auto out_shape = to_shape(*dest_shape_ptr);

        const auto layer_name = prefix + std::to_string(out_idx);

        auto* in_ptr = op.inputs();
        assert(in_ptr && in_ptr->size() > 0);
        std::vector<std::string> in_names;
        std::vector<shape_type> in_shapes;
        std::vector<dlib::tensor const*> in_params;
        for (auto idx : *in_ptr) {
            assert(0 <= idx && unsigned(idx) < sg_tensors.size());
            auto& t = tensors[unsigned(idx)];
            if (t.size() > 0)
                in_params.push_back(&t);
            else {
                // data input -- note: idx == 0 is net input
                auto name = prefix + std::to_string(idx);
                if (layer_names.empty()) {
                    if (idx != 0) FILE_LOG(logERROR) << "idx = " << idx;
                    //assert(idx == 0);
                }
                else
                    assert(layer_names.count(name) == 1);
                in_names.push_back(move(name));
                auto* src_ptr = sg_tensors[unsigned(idx)];
                assert(src_ptr);
                tflite::Tensor const& src = *src_ptr;
                auto* src_shape_ptr = src.shape();
                assert(src_shape_ptr);
                in_shapes.push_back(to_shape(*src_shape_ptr));
            }
        }
        assert(layer_names.empty() || 0 < in_names.size());

        //FILE_LOG(logINFO) << layer_name;
        const layer_args args {
            out_shape,
            in_shapes,
            in_params,
            op
        };

        dest_layers.emplace_back(make_layer(opcode, args));
        dest_layers.back()->name = layer_name;
        dest_layers.back()->inbound = in_names;

        const auto pr = layer_names.insert(layer_name);
        assert(pr.second);
    }

    // remove identity layers
    map_layers(dest_layers.begin(), dest_layers.end());
    for (unsigned i = 0; i < dest_layers.size(); ) {
        auto& layer = *dest_layers[i];
        using T = dlibx::net::layer_generic<dlibx::transpose_>;
        if (auto ptr = dynamic_cast<T const*>(&layer)) {
            auto& detail = ptr->detail;
            if (detail.mode() == dlibx::TRANSPOSE_KRC &&
                detail.k() == 0 && detail.nr() == 0 && detail.nc() == 0) {
                assert(ptr->inbound.size() == 1 &&
                       ptr->inbound_nodes.size() == 1 &&
                       ptr->inbound_nodes.front() != nullptr);
                auto& inbound = *ptr->inbound_nodes.front();
                if (!ptr->outbound_nodes.empty() ||
                    inbound.outbound_nodes.size() == 1) {
                    FILE_LOG(logINFO) << "remove identity layer";
                    for (auto* out : ptr->outbound_nodes)
                        for (auto& x : out->inbound)
                            if (x == ptr->name)
                                x = ptr->inbound.front();
                    dest_layers.erase(next(dest_layers.begin(), i));
                    continue; // layer was deleted
                }
                else
                    FILE_LOG(logWARNING)
                        << "not removing identity output layer";
            }
        }
        if (layer.outbound_nodes.empty())
            FILE_LOG(logINFO) << "output layer: " << layer.name
                              << ' ' << i+1 << " / " << dest_layers.size();
        ++i;
    }

    // remove last output
    if (1) {
        map_layers(dest_layers.begin(), dest_layers.end());
        auto n = remove_output(dest_layers, dest_layers.back().get());
        FILE_LOG(logINFO) << "remove last layer removed " << n << " layers";
    }

    // remove z-coord from mesh478 output
    if (1) {
        bool found = false;
        for (auto& uptr : dest_layers) {
            using T = dlibx::net::layer_generic<dlibx::lm_con_<1,2,2,1,1,0,0> >;
            if (auto* ptr = dynamic_cast<T*>(uptr.get()))
                if (ptr->detail.num_filters() == 1434) {
                    output_subset(ptr->detail, 2, 3);
                    found = true;
                }
        }
        assert(found);
    }

    // create net::vector
    auto nv = dlibx::net::vector(move(dest_layers));
    if (0 < input_dim) {
        auto s = "retina" + std::to_string(input_dim) + "*2.85+0.35rgb";
        nv.input_extractor = raw_image::input_extractor::find(s);
    }
    dlib::serialize(nv_path.c_str()) << nv;

    // create 68 point subset
    if (1) {
        auto nv68 = nv;
        auto layers = nv68.release_layers();
        bool found = false;
        for (auto& uptr : layers) {
            if (auto* ptr = dynamic_cast<dlibx::net::layer_con*>(uptr.get())) {
                //FILE_LOG(logINFO) << "con: " << ptr->num_filters() << ' ' << ptr->code() << ' ' << typeid(*ptr).name();
            }
            using T = dlibx::net::layer_con_t<dlibx::lm_con_<1,2,2,1,1,0,0> >;
            if (auto* ptr = dynamic_cast<T*>(uptr.get())) {
                FILE_LOG(logINFO) << ptr->detail.num_filters();
                if (ptr->detail.num_filters() == 2*478) {
                    using dt = raw_image::detection_type;
                    auto lmidx =
                        raw_image::landmark_subset(dt::mesh478, dt::dlib68);
                    assert(lmidx.size() == 68);
                    std::vector<unsigned> coordidx;
                    coordidx.reserve(lmidx.size()*2);
                    for (auto idx : lmidx) {
                        coordidx.push_back(2*idx);
                        coordidx.push_back(2*idx+1);
                    }
                    assert(coordidx.size() == 2*68);
                    output_subset(ptr->detail, coordidx);
                    found = true;
                }
            }
        }
        assert(found);
        nv68.set_layers(move(layers));
        const auto path = nv_path.parent_path() /
            (nv_path.stem().generic_string() + "-68.nv");
        dlib::serialize(path.c_str()) << nv68;
    }

    FILE_LOG(logINFO) << "--";
    
    // load sample image
    auto raw = load("1037.png", raw_image::pixel::rgb24);
    //auto raw = load("IMG_20240530_172120c.jpg", raw_image::pixel::rgb24);

    assert(nv.input_extractor);
    if (raw->width != input_dim || raw->height != input_dim) {
        FILE_LOG(logINFO) << "resize from " << raw->width << 'x' << raw->height
                          << " to " << input_dim << 'x' << input_dim;
        raw = copy_resize(raw, input_dim, input_dim);
    }

    // tflite inference
    auto tflite_out = tflite::infer(tflite_model.fbmodel, *raw);
    FILE_LOG(logINFO) << "tflite outputs: " << tflite_out.size();
    for (dlib::tensor const& t : tflite_out) {
        FILE_LOG(logINFO) << '\t' << to_shape(t) << '\t' << *t.host();
    }

    // nv inference
    std::vector<dlib::resizable_tensor> nv_out(8);
    if (auto n = nv(raw, nv_out)) {
        FILE_LOG(logINFO) << "nv outputs: " << n;
        assert(n <= nv_out.size());
        nv_out.resize(n);
    }
    else throw std::runtime_error("no output");
    for (dlib::tensor const& t : nv_out) {
        FILE_LOG(logINFO) << '\t' << to_shape(t) << '\t' << *t.host();
    }

    FILE_LOG(logINFO) << "--";
    for (auto& ptr : nv) {
        if (ptr.outbound_nodes.empty())
            FILE_LOG(logINFO) << ptr.concise();
    }

    FILE_LOG(logINFO) << "--";
    return 0;
}
