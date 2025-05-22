
#include <dlibx/net_vector.hpp>
#include <dlibx/net_layer.hpp>
#include <dlibx/net_layer_impl.hpp>
#include <dlibx/input_extractor.hpp>
#include <dlibx/library_init.hpp>

#include <applog/core.hpp>

#include <tensorflow/lite/model_builder.h>
#include <tensorflow/lite/kernels/register.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>


/*
namespace {
    struct layer_param {
        std::string type, name;
        std::vector<std::string> inputs, outputs;
        std::vector<std::pair<int,float> > params;
    };

    struct weight_buffer {
        float const* data;
        std::size_t size;
        long flag = -1;  // flag omitted if < 0, 0=float32
        dlib::resizable_tensor t;

        weight_buffer(const dlib::tensor& t)
            : data(t.host()), size(t.size()) {
        }
        weight_buffer(const dlib::tensor& t, std::size_t size)
            : data(t.host()), size(size) {
            assert(size <= t.size());
        }
        weight_buffer(const dlib::tensor& t, std::size_t ofs, std::size_t size)
            : data(t.host()+ofs), size(size) {
            assert(ofs+size <= t.size());
        }
    };
}
*/


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
                           << '\t' << prog << " model_file.nv";
        return 1;
    }
    const auto model_filename = std::filesystem::path(*argv);
    ++argv, --argc;
    if (!is_regular_file(model_filename)) {
        FILE_LOG(logFATAL) << "file not found: " << model_filename;
        return 1;
    }

    const auto tflite_filename =
        std::filesystem::path(model_filename).filename().replace_extension(".tflite");
    if (exists(tflite_filename)) {
        FILE_LOG(logERROR) << "destination path exists:" << tflite_filename;
        return 1;
    }

    // library init
    dlibx::library_init();

    // load model
    auto src_model =
        dlibx::net::vector(std::ifstream(model_filename,std::ios::binary));
    assert(!src_model.empty());
    auto src = src_model.release_layers();
    FILE_LOG(logINFO) << "layers: " << src.size();
    assert(0 < src.size());


    //using FlatBufferModel = ::tflite::FlatBufferModel;
    //std::unique_ptr<const FlatBufferModel> dest_model;

    //tflite::Model m;
    // subgraph
    
    /*
    std::vector<layer_param> dest;
    dest.reserve(src.size());
    std::set<std::string> blob_names;
    unsigned extra_blob_counter = 0;
    const auto extra_blob = [&]() {
        for (;;) {
            auto name = "xblob" + std::to_string(extra_blob_counter++);
            if (blob_names.count(name) == 0) {
                blob_names.insert(name);
                return name;
            }
        }
    };
    unsigned split_counter = 0;

    std::vector<weight_buffer> weights;
    weights.reserve(src.size());
    
    // conversion
    for (const auto& ptr : src) {
        auto& rec = dest.emplace_back();
        rec.name = ptr->name;
        rec.inputs = ptr->inbound;
        for (auto& name : rec.inputs)
            assert(blob_names.count(name) == 1);

        assert(blob_names.count(rec.name) == 0);
        blob_names.insert(rec.name);
        rec.outputs.push_back(rec.name);

        if (1 < ptr->outbound_nodes.size()) {
            // need split layer
            auto& split = dest.emplace_back();
            split.type = "Split";
            split.name = "splitncnn_" + std::to_string(split_counter++);
            split.inputs = { rec.name };
            for (auto* p : ptr->outbound_nodes) {
                auto name = extra_blob();
                bool found = false;
                for (auto& x : p->inbound)
                    if (x == rec.name) {
                        x = name;
                        found = true;
                        break;
                    }
                assert(found);
                split.outputs.push_back(name);
            }
        }

        using namespace dlibx::net;
        using layer_prelu = layer_generic<dlibx::prelu_>;
        using layer_sig = layer_generic<dlib::sig_>;
        using layer_avg_pool_all = layer_generic<dlib::avg_pool_<0,0,1,1> >;

        if (const auto* p = dynamic_cast<layer_relu*>(ptr.get())) {
            rec.type = "ReLU";
            (void)p; // no parameters required
        }

        else if (const auto* p = dynamic_cast<layer_prelu*>(ptr.get())) {
            rec.type = "PReLU";
            auto&& t = p->get_layer_params();
            rec.params.emplace_back(0, t.size()); // num_slope
            weights.emplace_back(t);
        }

        else if (const auto* p = dynamic_cast<layer_sig*>(ptr.get())) {
            rec.type = "Sigmoid";
            (void)p; // no parameters required
        }

        else if (const auto* p = dynamic_cast<layer_add_prev*>(ptr.get())) {
            rec.type = "BinaryOp";
            (void)p; // no parameters required -- default is Operation_ADD
        }

        else if (const auto* p = dynamic_cast<layer_avg_pool_all*>(ptr.get())) {
            (void)p;
            rec.type = "Pooling";

            // pooling_type = PoolMethod_AVE
            rec.params.emplace_back(0, 1);

            // global_pooling = pd.get(4, 0);
            rec.params.emplace_back(4, 1);

            // none of the other params matter when global pooling
        }

        else if (const auto* p = dynamic_cast<layer_fc*>(ptr.get())) {
            rec.type = "InnerProduct";

            const auto k = p->get_num_outputs();
            rec.params.emplace_back(0, k);
            rec.params.emplace_back(1, p->has_bias());

            auto&& t = p->get_layer_params();
            auto n = t.size();
            if (p->has_bias())
                n -= k;
            rec.params.emplace_back(2, n); // weight_data_size

            {
                // transpose weights
                const dlib::alias_tensor at(long(n/k), long(k));
                auto ac = at(t, 0);
                dlib::tensor const& a = ac;
                auto r = dlib::resizable_tensor(long(k), long(n/k));
                r = trans(mat(a));
                auto& rec = weights.emplace_back(r, n);
                rec.flag = 0; // weights
                rec.t = std::move(r);
            }
            if (p->has_bias())
                weights.emplace_back(t, n, k); // bias
        }

        else if (const auto* p = dynamic_cast<layer_con*>(ptr.get())) {
            // num_output
            const auto num_output = p->num_filters();
            rec.params.emplace_back(0, num_output);

            // kernel_w and kernel_h
            const auto filter_pixels = unsigned(p->nc()*p->nr());
            rec.params.emplace_back(1, p->nc());
            if (p->nr() != p->nc())
                rec.params.emplace_back(11, p->nr());

            // stride
            const auto stride = p->stride();
            if (stride[0] != 1)
                rec.params.emplace_back(3, stride[0]);
            if (stride[1] != stride[0])
                rec.params.emplace_back(13, stride[1]);

            // padding
            const auto pad = p->padding();
            if (pad[0])
                rec.params.emplace_back(4, pad[0]); // pad_left and right
            if (pad[1] != pad[0])
                rec.params.emplace_back(14, pad[0]); // pad_top and bottom

            // bias
            rec.params.emplace_back(5, p->has_bias());

            // weight_data_size
            auto&& t = p->get_layer_params();
            auto weight_data_size = t.size();
            if (p->has_bias())
                weight_data_size -= unsigned(p->num_filters());
            rec.params.emplace_back(6, weight_data_size);

            // weights
            weights.emplace_back(t, weight_data_size).flag = 0; // weights
            if (p->has_bias())
                weights.emplace_back(t, weight_data_size, p->num_filters());

            const auto type = p->code().substr(0,4);
            if (type == "con_") {
                rec.type = "Convolution";
            }

            else if (type == "cdw_") {
                rec.type = "ConvolutionDepthWise";
                assert(weight_data_size == unsigned(num_output) * filter_pixels);
                // group = num_inputs <= num_filters
                // get_depth_multiplier() !! NOT RIGHT !! if multiplier != 1
                rec.params.emplace_back(7, p->num_filters());
            }

            else {
                FILE_LOG(logERROR) << "unknown convolution: " << ptr->code();
                throw std::runtime_error("convolution not handled");
            }
        }

        else if (const auto* p = dynamic_cast<layer_input<dlibx::input_generic_image<dlib::matrix<dlib::rgb_pixel> > >*>(ptr.get())) {
            rec.type = "Input";
            float ofs, scale;
            using norm_type = dlibx::input_normalization;
            switch (p->detail.get_input_normalization()) {
            case norm_type::none:        // default range [0,1]
                ofs = 0, scale = 1.0f / 256;
                break;
            case norm_type::zero_center: // output range is [-1,1]
                ofs = 127.5f, scale = 1.0f / 128;
                break;
            case norm_type::minmax: // extend values to fill range
            case norm_type::minmax_zero_center:
            default:
                throw std::runtime_error(
                    "input normalization method not handled");
            }
            if (ofs < 0 || 0 < ofs) {
                auto& prev = dest.back();
                auto& op = dest.emplace_back();
                op.type = "BinaryOp";
                op.name = extra_blob();
                op.inputs = { op.name };
                op.outputs = prev.outputs;
                prev.outputs = op.inputs;
                op.params.emplace_back(0, 1); // Operation_SUB
                op.params.emplace_back(1, 1); // with_scalar
                op.params.emplace_back(2, ofs);
            }
            if (0 < scale) {
                auto& prev = dest.back();
                auto& op = dest.emplace_back();
                op.type = "BinaryOp";
                op.name = extra_blob();
                op.inputs = { op.name };
                op.outputs = prev.outputs;
                prev.outputs = op.inputs;
                op.params.emplace_back(0, 2); // Operation_MUL
                op.params.emplace_back(1, 1); // with_scalar
                op.params.emplace_back(2, scale);
            }
        }

        else {
            FILE_LOG(logERROR) << "layer not handled: " << ptr->code();
            throw std::runtime_error("layer not handled");
        }
    }

    // optionally change output name
    static constexpr auto output_name = "output";
    assert(dest.back().outputs.size() == 1);
    if (dest.back().outputs.front() != output_name) {
        assert(blob_names.count(output_name) == 0);
        blob_names.erase(dest.back().outputs.front());
        blob_names.insert(output_name);
        dest.back().outputs = { output_name };
    }

    {
        // widths
        int width_type = 24, width_name = 24;
        for (auto& rec : dest) {
            width_type = std::max(width_type, int(rec.type.size()));
            width_name = std::max(width_name, int(rec.name.size()));
        }

        // write param file
        auto param_out = std::ofstream(param_filename);
        param_out << 7767517 << std::endl
                  << dest.size() << ' '
                  << blob_names.size() << std::endl;
        param_out << std::left;
        for (auto& rec : dest) {
            param_out << std::setw(width_type) << rec.type << ' '
                      << std::setw(width_name) << rec.name << ' '
                      << rec.inputs.size() << ' ' << rec.outputs.size();
            for (const auto& name : rec.inputs)
                param_out << ' ' << name;
            for (const auto& name : rec.outputs)
                param_out << ' ' << name;
            for (const auto& pr : rec.params)
                param_out << ' ' << pr.first << '=' << pr.second;
            param_out << std::endl;
        }
    }

    {
        // write weights file
        auto bin_out = std::ofstream(bin_filename, std::ios::binary);
        for (auto& rec : weights) {
            if (0 <= rec.flag) {
                auto flag = uint32_t(rec.flag);
                bin_out.write(reinterpret_cast<char const*>(&flag), 4);
            }
            bin_out.write(reinterpret_cast<char const*>(rec.data),
                          4 * long(rec.size));
        }
    }
    */
    
    return 0;
}
