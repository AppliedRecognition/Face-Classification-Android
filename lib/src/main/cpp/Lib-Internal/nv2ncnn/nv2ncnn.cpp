
#include <dlibx/net_vector.hpp>
#include <dlibx/net_layer.hpp>
#include <dlibx/net_layer_impl.hpp>
#include <dlibx/input_extractor.hpp>
#include <dlibx/library_init.hpp>

#include <applog/core.hpp>

// ncnn
#include <mat.h>
#include <net.h>
#include <datareader.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>


namespace {
    struct layer_param {
        std::string type, name;
        std::vector<std::string> inputs, outputs;
        std::vector<std::pair<int,std::string> > params;
    };

    struct weight_buffer {
        float const* data;
        std::size_t size;
        long flag = -1;  // flag omitted if < 0, 0=float32
        dlib::resizable_tensor buf;

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

    const auto model_path = std::filesystem::path(*argv);
    ++argv, --argc;
    if (!is_regular_file(model_path)) {
        FILE_LOG(logFATAL) << "file not found: " << model_path;
        return 1;
    }

    const auto param_path = model_path.filename().replace_extension(".param");
    const auto bin_path =   model_path.filename().replace_extension(".bin");
    if (exists(param_path) || exists(bin_path)) {
        FILE_LOG(logERROR) << "destination path exists:" << std::endl
                           << '\t' << param_path << " or " << std::endl
                           << '\t' << bin_path;
        return 1;
    }

    // library init
    dlibx::library_init();

    // load model
    auto model =
        dlibx::net::vector(std::ifstream(model_path,std::ios::binary));
    assert(!model.empty());
    if (model.input_extractor) {
        FILE_LOG(logINFO) << "input extractor: " << model.input_extractor->name;
        const auto w = model.input_extractor->width;
        const auto h = model.input_extractor->height;
        auto raw = create(w, h, raw_image::pixel::rgb24);
        model(raw);
    }
    auto src = model.release_layers();
    FILE_LOG(logINFO) << "layers: " << src.size();
    assert(0 < src.size());

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

    std::map<std::string, std::string> input_replace;

    // conversion
    for (const auto& ptr : src) {
        auto& rec = dest.emplace_back();
        rec.name = ptr->name;
        for (auto& s : ptr->inbound) {
            const auto it = input_replace.find(s);
            rec.inputs.push_back(it != input_replace.end() ? it->second : s);
        }
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
        using layer_max_pool_2_2 = dlibx::net::layer_generic<dlib::max_pool_<2,2,2,2> >;

        if (const auto* p = dynamic_cast<layer_relu*>(ptr.get())) {
            rec.type = "ReLU";
            (void)p; // no parameters required
        }

        else if (const auto* p = dynamic_cast<layer_prelu*>(ptr.get())) {
            rec.type = "PReLU";
            auto&& t = p->get_layer_params();
            rec.params.emplace_back(0, std::to_string(t.size())); // num_slope
            weights.emplace_back(t);
        }

        else if (const auto* p = dynamic_cast<layer_sig*>(ptr.get())) {
            rec.type = "Sigmoid";
            (void)p; // no parameters required
        }

        else if (const auto* p = dynamic_cast<layer_add_prev*>(ptr.get())) {
            rec.type = "BinaryOp";  // default is Operation_ADD

            // check layer sizes
            assert(p->inbound_nodes.size() >= 2);
            unsigned k = 0, nr = 0, nc = 0;
            for (auto* in : p->inbound_nodes) {
                auto& t = in->last_output();
                assert(t.num_samples() == 1 &&
                       0 < t.k() && 0 < t.nr() && 0 < t.nc());
                k  = std::max(k,  unsigned(t.k()));
                nr = std::max(nr, unsigned(t.nr()));
                nc = std::max(nc, unsigned(t.nc()));
            }

            // add padding layers if necessary
            unsigned idx = 0;
            for (auto* in : p->inbound_nodes) {
                auto& t = in->last_output();
                if (k != t.k() || nr != t.nr() || nc != t.nc()) {
                    FILE_LOG(logINFO) << "padding: "
                                      << t.k() << 'x' << t.nr() << 'x' << t.nc()
                                      << " -> " << k << 'x' << nr << 'x' << nc;
                    auto it = dest.emplace(prev(dest.end()));
                    auto& pad = *it;
                    auto& rec = *next(it);
                    pad.type = "Padding";
                    pad.name = extra_blob();
                    pad.outputs = { pad.name };
                    pad.inputs = { rec.inputs[idx] };
                    rec.inputs[idx] = pad.name;
                    if (nr != t.nr()) // bottom
                        pad.params.emplace_back(1, std::to_string(nr - t.nr()));
                    if (nc != t.nc()) // right
                        pad.params.emplace_back(3, std::to_string(nc - t.nc()));
                    if (k != t.k()) // behind
                        pad.params.emplace_back(8, std::to_string(k - t.k()));
                }
                ++idx;
            }
        }

        else if (const auto* p = dynamic_cast<layer_max_pool_2_2*>(ptr.get())) {
            (void)p;
            rec.type = "Pooling";

            // pooling_type = PoolMethod_MAX
            rec.params.emplace_back(0, "0"); // default

            rec.params.emplace_back(1, "2"); // kernel w and h
            rec.params.emplace_back(2, "2"); // stride w and h
        }

        else if (const auto* p = dynamic_cast<layer_avg_pool_all*>(ptr.get())) {
            (void)p;
            rec.type = "Pooling";

            // pooling_type = PoolMethod_AVE
            rec.params.emplace_back(0, "1");

            // global_pooling = pd.get(4, 0);
            rec.params.emplace_back(4, "1");

            // none of the other params matter when global pooling
        }

        else if (const auto* p = dynamic_cast<layer_fc*>(ptr.get())) {
            rec.type = "InnerProduct";

            const auto k = p->get_num_outputs();
            rec.params.emplace_back(0, std::to_string(k));
            if (p->has_bias())
                rec.params.emplace_back(1, "1");

            auto&& t = p->get_layer_params();
            auto n = t.size();
            if (p->has_bias())
                n -= k;
            rec.params.emplace_back(2, std::to_string(n)); // weight_data_size

            {
                // transpose weights
                const dlib::alias_tensor at(long(n/k), long(k));
                auto ac = at(t, 0);
                dlib::tensor const& a = ac;
                auto r = dlib::resizable_tensor(long(k), long(n/k));
                r = trans(mat(a));
                auto& w = weights.emplace_back(r, n);
                w.flag = 0; // weights
                w.buf = std::move(r);
            }
            if (p->has_bias())
                weights.emplace_back(t, n, k); // bias
        }

        else if (const auto* p = dynamic_cast<layer_con*>(ptr.get())) {
            // num_output
            const auto num_output = p->num_filters();
            rec.params.emplace_back(0, std::to_string(num_output));

            // kernel_w and kernel_h
            const auto filter_pixels = unsigned(p->nc()*p->nr());
            rec.params.emplace_back(1, std::to_string(p->nc()));
            if (p->nr() != p->nc())
                rec.params.emplace_back(11, std::to_string(p->nr()));

            // stride
            const auto stride = p->stride();
            if (stride[0] != 1)
                rec.params.emplace_back(3, std::to_string(stride[0]));
            if (stride[1] != stride[0])
                rec.params.emplace_back(13, std::to_string(stride[1]));

            // padding
            const auto pad = p->padding();
            if (pad[0])
                rec.params.emplace_back(4, std::to_string(pad[0])); // pad_left and right
            if (pad[1] != pad[0])
                rec.params.emplace_back(14, std::to_string(pad[1])); // pad_top and bottom

            // bias
            if (p->has_bias())
                rec.params.emplace_back(5, "1");

            // weight_data_size
            auto&& t = p->get_layer_params();
            auto weight_data_size = t.size();
            if (p->has_bias())
                weight_data_size -= unsigned(p->num_filters());
            rec.params.emplace_back(6, std::to_string(weight_data_size));

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
                rec.params.emplace_back(7, std::to_string(p->num_filters()));
            }

            else {
                FILE_LOG(logERROR) << "unknown convolution: " << ptr->code();
                throw std::runtime_error("convolution not handled");
            }
        }

        else if (const auto* p = dynamic_cast<layer_input<dlibx::input_generic_image<dlib::matrix<dlib::rgb_pixel> > >*>(ptr.get())) {
            rec.type = "Input";
            rec.name = "input";
            assert(rec.inputs.empty() && rec.outputs.size() == 1);
            float ofs, scale;
            using norm_type = dlibx::input_normalization;
            switch (p->detail.get_input_normalization()) {
            case norm_type::none:        // default range [0,1]
                ofs = 0, scale = 1.0f / 256;
                break;
            case norm_type::zero_center: // output range is [-1,1]
                ofs = 128, scale = 1.0f / 128;
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
                op.inputs = prev.outputs;
                op.outputs = { op.name };
                prev.outputs = op.inputs;
                op.params.emplace_back(0, "1"); // Operation_SUB
                op.params.emplace_back(1, "1"); // with_scalar
                std::stringstream ss;
                ss << std::scientific << ofs;
                op.params.emplace_back(2, ss.str());
                input_replace[rec.outputs[0]] = op.name;
            }
            if (0 < scale) {
                auto& prev = dest.back();
                auto& op = dest.emplace_back();
                op.type = "BinaryOp";
                op.name = extra_blob();
                op.inputs = prev.outputs;
                op.outputs = { op.name };
                prev.outputs = op.inputs;
                op.params.emplace_back(0, "2"); // Operation_MUL
                op.params.emplace_back(1, "1"); // with_scalar
                std::stringstream ss;
                ss << std::scientific << scale;
                op.params.emplace_back(2, ss.str());
                input_replace[rec.outputs[0]] = op.name;
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
        auto param_out = std::ofstream(param_path);
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
        auto bin_out = std::ofstream(bin_path, std::ios::binary);
        for (auto& rec : weights) {
            if (0 <= rec.flag) {
                auto flag = uint32_t(rec.flag);
                bin_out.write(reinterpret_cast<char const*>(&flag), 4);
            }
            bin_out.write(reinterpret_cast<char const*>(rec.data),
                          4 * long(rec.size));
        }
    }
    
    return 0;
}
