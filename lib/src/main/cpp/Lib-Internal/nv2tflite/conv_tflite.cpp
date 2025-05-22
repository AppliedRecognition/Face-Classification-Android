
#include "conv_tflite.hpp"
#include "float16.hpp"

#include <dlibx/net_layer_impl.hpp>

#include <applog/core.hpp>


using namespace conv;

static const auto float16_good = []() {
    if (!(std::abs(to_float(float16_t(0))) <= 0))
        return false;
    if (!(std::abs(to_float(float16_t(0x3c00)) - 1) <= 0))
        return false;
    if (!(std::abs(to_float(float16_t(0x7bff)) - 65504) <= 0))
        return false;
    if (!std::isinf(to_float(float16_t(0x7c00))))
        return false;
    if (!std::isnan(to_float(float16_t(0x7c01))))
        return false;
    if (!std::isnan(to_float(float16_t(0x7cff))))
        return false;
    float x = -1;
    for (unsigned i = 0; i < 0x7c00; ++i) {
        auto y = to_float(float16_t(i));
        if (!(x < y)) return false;
        x = y;
        auto z = to_float(float16_t(0x8000 + i));
        if (!(std::abs(y+z) <= 0))
            return false;
    }
    return true;
}();


void conv::dequantize(const flatbuffers::Vector<uint8_t>& src_data,
                      tflite::TensorType src_type, unsigned bytes_per_el,
                      const shape_type& shape, dlib::resizable_tensor& dest) {

    dest.set_size(shape[0],shape[1],shape[2],shape[3]);
    float* dp = dest.host_write_only();

    switch (src_type) {
    case tflite::TensorType_FLOAT16: {
        assert(float16_good);
        assert(bytes_per_el == 2);
        auto* sp = reinterpret_cast<float16_t const*>(src_data.data());
        for (auto n = src_data.size() / bytes_per_el; n > 0; --n, ++sp, ++dp)
            *dp = to_float(*sp);
        break;
    }
    default:
        throw std::runtime_error("unsupported source type for dequantize");
    }
}

static dlibx::net::layer_ptr
make_layer_conv2d(const layer_args& args) {
    assert(args.in_shapes.size() == 1 && args.params.size() == 2);
    auto& in_shape = args.in_shapes.front();
    FILE_LOG(logDETAIL) << "con: " << args.out_shape << " <- " << in_shape
                        << ' ' << to_shape(*args.params[0])
                        << ' ' << to_shape(*args.params[1]);

    assert(args.out_shape[0] == 1);
    assert(in_shape[0] == 1);
    const auto k_in = in_shape[3];
    const auto k_out = args.out_shape[3];

    auto& weights = *args.params.front();
    assert(weights.num_samples() == k_out);
    const auto nr = weights.k();
    const auto nc = weights.nr();
    assert(weights.nc() == k_in);

    auto& biases = *args.params.back();
    assert(biases.size() == k_out);

    dlib::resizable_tensor input(
        in_shape[0],in_shape[3],in_shape[1],in_shape[2]);
    struct sub {
        const dlib::tensor& t;
        auto& get_output() const { return t; }
    };

    const auto make = [&](auto con) {
        using T = decltype(con);
        auto ptr = std::make_unique<dlibx::net::layer_generic<T> >(con);
        auto& detail = ptr->detail;
        detail.setup(sub{input});

        dlib::resizable_tensor output;
        detail.forward(sub{input},output);
        assert(output.num_samples() == 1);
        assert(output.k() == k_out);
        assert(output.nr() == args.out_shape[1]);
        assert(output.nc() == args.out_shape[2]);

        auto& params = detail.get_layer_params();
        assert(params.size() == weights.size() + biases.size());

        const auto src = rotate(weights);
        float* dest = params.host_write_only();
        dest = std::copy_n(src.host(), src.size(), dest);
        std::copy_n(biases.host(), biases.size(), dest);

        return ptr;
    };

    auto opts_ptr = args.op.builtin_options_as_Conv2DOptions();
    assert(opts_ptr);
    auto& opts = *opts_ptr;
    assert(opts.dilation_w_factor() == 1 && opts.dilation_h_factor() == 1);

    assert(opts.stride_w() == opts.stride_h() && 1 <= opts.stride_w());
    const auto stride = unsigned(opts.stride_w());

    assert(0 < nr && nr < 10);
    assert(0 < nc && nc < 10);
    assert(0 < stride && stride < 10);
    assert(0 <= opts.padding() && opts.padding() < 2);

    const auto combo = nr*1000 + nc*100 + stride*10 + (1-opts.padding());
    switch (combo) {
    case 1110:
    case 1111: return make(dlibx::lm_con_<1,1,1,1,1>(k_out));

    case 2210: return make(dlibx::lm_con_<1,2,2,1,1,0,0>(k_out));
    case 2220: return make(dlibx::lm_con_<1,2,2,2,2,0,0>(k_out));

    case 3310: return make(dlibx::lm_con_<1,3,3,1,1,0,0>(k_out));
    case 3311: return make(dlibx::lm_con_<1,3,3,1,1,1,1>(k_out));
    case 3320: return make(dlibx::lm_con_<1,3,3,2,2,0,0>(k_out));
    case 3321: return make(dlibx::lm_con_<1,3,3,2,2,1,1>(k_out));

    case 5521: return make(dlibx::lm_con_<1,5,5,2,2,2,2>(k_out));
    }
    FILE_LOG(logINFO) << "con: " << args.out_shape << " <- " << in_shape
                      << ' ' << to_shape(*args.params[0])
                      << ' ' << to_shape(*args.params[1]);
    FILE_LOG(logERROR) << "unsupported conv2d combo " << combo
                       << ' ' << nr << 'x' << nc << " stride " << stride
                       << " padding " << int(opts.padding());
    throw std::runtime_error("conv2d not implemented");
}

static dlibx::net::layer_ptr
make_layer_dconv2d(const layer_args& args) {
    assert(args.in_shapes.size() == 1 && args.params.size() == 2);
    auto& in_shape = args.in_shapes.front();
    FILE_LOG(logDETAIL) << "cdw: " << args.out_shape << " <- " << in_shape
                        << ' ' << to_shape(*args.params[0])
                        << ' ' << to_shape(*args.params[1]);

    assert(args.out_shape[0] == 1);
    assert(in_shape[0] == 1);
    const auto k = in_shape[3];
    assert(args.out_shape[3] == k);

    auto& weights = *args.params.front();
    assert(weights.num_samples() == 1);
    const auto nr = weights.k();
    const auto nc = weights.nr();
    assert(weights.nc() == k);

    auto& biases = *args.params.back();
    assert(biases.size() == k);

    dlib::resizable_tensor input(
        in_shape[0],in_shape[3],in_shape[1],in_shape[2]);
    struct sub {
        const dlib::tensor& t;
        auto& get_output() const { return t; }
    };

    const auto make = [&](auto con) {
        using T = decltype(con);
        auto ptr = std::make_unique<dlibx::net::layer_generic<T> >(con);
        auto& detail = ptr->detail;
        detail.setup(sub{input});

        dlib::resizable_tensor output;
        detail.forward(sub{input},output);
        assert(output.num_samples() == 1);
        assert(output.k() == k);
        assert(output.nr() == args.out_shape[1]);
        assert(output.nc() == args.out_shape[2]);

        auto& params = detail.get_layer_params();
        assert(params.size() == weights.size() + biases.size());

        const auto src = rotate(weights);
        float* dest = params.host_write_only();
        dest = std::copy_n(src.host(), src.size(), dest);
        std::copy_n(biases.host(), biases.size(), dest);

        return ptr;
    };

    auto opts_ptr = args.op.builtin_options_as_DepthwiseConv2DOptions();
    assert(opts_ptr);
    auto& opts = *opts_ptr;
    assert(opts.depth_multiplier() == 1);
    assert(opts.dilation_w_factor() == 1 && opts.dilation_h_factor() == 1);

    assert(opts.stride_w() == opts.stride_h() && 1 <= opts.stride_w());
    const auto stride = unsigned(opts.stride_w());

    assert(0 < nr && nr < 10);
    assert(0 < nc && nc < 10);
    assert(0 < stride && stride < 10);
    assert(0 <= opts.padding() && opts.padding() < 2);

    const auto combo = nr*1000 + nc*100 + stride*10 + (1-opts.padding());
    switch (combo) {
    case 3310: return make(dlibx::condw_<dlibx::HAS_BIAS,1,3,3,1,1,0,0>());
    case 3311: return make(dlibx::condw_<dlibx::HAS_BIAS,1,3,3,1,1,1,1>());
    case 3320: return make(dlibx::condw_<dlibx::HAS_BIAS,1,3,3,2,2,0,0>());
    case 3321: return make(dlibx::condw_<dlibx::HAS_BIAS,1,3,3,2,2,1,1>());
    }
    FILE_LOG(logINFO) << "cdw: " << args.out_shape << " <- " << in_shape
                      << ' ' << to_shape(*args.params[0])
                      << ' ' << to_shape(*args.params[1]);
    FILE_LOG(logERROR) << "unsupported dwconv2d combo " << combo
                       << ' ' << nr << 'x' << nc << " stride " << stride
                       << " padding " << int(opts.padding());
    throw std::runtime_error("dwconv2d not implemented");
}

static dlibx::net::layer_ptr
make_layer_maxpool2d(const layer_args& args) {
    assert(args.in_shapes.size() == 1 && args.params.size() == 0);
    auto& in_shape = args.in_shapes.front();

    auto opts_ptr = args.op.builtin_options_as_Pool2DOptions();
    assert(opts_ptr);
    auto& opts = *opts_ptr;

    FILE_LOG(logINFO) << "maxpool: "
                      << opts.stride_w() << 'x' << opts.stride_h() << ' '
                      << opts.filter_width() << 'x' << opts.filter_height() << ' '
                      << in_shape << " -> " << args.out_shape
                      << " pad " << tflite::EnumNamesPadding()[opts.padding()];

    const auto stride = in_shape[1] / args.out_shape[1];
    assert(in_shape[0] == 1 && args.out_shape[0] == 1);
    assert(in_shape[1] == stride * args.out_shape[1]);
    assert(in_shape[2] == stride * args.out_shape[2]);
    assert(in_shape[3] == args.out_shape[3]);

    assert(int(stride) == opts.stride_w());
    assert(int(stride) == opts.stride_h());

    switch (stride) {
    case 2:
        if (opts.filter_height() == 2 && opts.filter_width() == 2)
            return std::make_unique<
                dlibx::net::layer_generic<dlib::max_pool_<2,2,2,2> > >();
        break;

    default:
        throw std::runtime_error("unsupported max_pool stride");
    }
    throw std::runtime_error("unsupported max_pool kernel");
}

static dlibx::net::layer_ptr
make_layer_pad(const layer_args& args) {
    assert(args.in_shapes.size() == 1 && args.params.size() == 1);
    auto& in_shape = args.in_shapes.front();
    auto& p = *args.params.front();
    assert(p.k() == 2 && p.nr() == 1 && p.nc() == 1);
    json::array arr(p.begin(), p.end());

    FILE_LOG(logDETAIL) << "pad: " << json::value(arr) << ' '
                        << args.out_shape << " <- " << in_shape;

    const auto is_channel_append = [&]() -> std::optional<unsigned> {
        if (arr.size() != 8) return {};
        for (unsigned i = 0; i < 7; ++i)
            if (json::round_to<int>(arr[i]) != 0)
                return {};
        if (json::round_to<unsigned>(arr[7]) == 0)
            return {};
        return json::round_to<unsigned>(arr[7]);
    };
    if (auto o = is_channel_append()) {
        // add_prev will do this automatically, so identity
        assert(args.out_shape[0] == in_shape[0]);
        assert(args.out_shape[1] == in_shape[1]);
        assert(args.out_shape[2] == in_shape[2]);
        assert(args.out_shape[3] == in_shape[3] + *o);
        using T = dlibx::transpose_;
        auto ptr = std::make_unique<dlibx::net::layer_generic<T> >();
        return ptr;
    }
    
    const auto is_pad_equal = [&]() -> std::optional<unsigned> {
        if (arr.size() < 6) return {};
        const int n = round_from(arr[2]);
        if (n <= 0) return {};
        for (unsigned i = 0; i < arr.size(); ++i)
            if (json::round_to<int>(arr[i]) != (2 <= i && i < 6 ? n : 0))
                return {};
        return unsigned(n);
    };
    if (auto o = is_pad_equal()) {
        assert(args.out_shape[0] == in_shape[0]);
        assert(args.out_shape[1] == in_shape[1] + 2**o);
        assert(args.out_shape[2] == in_shape[2] + 2**o);
        assert(args.out_shape[3] == in_shape[3]);
        switch (*o) {
        case 1:
            return std::make_unique<
                dlibx::net::layer_generic<dlibx::padding_<1> > >();
        case 5:
            return std::make_unique<
                dlibx::net::layer_generic<dlibx::padding_<5> > >();
            
        default:
            throw std::runtime_error("unsupported padding size");
        }
    }

    FILE_LOG(logDETAIL) << "pad: " << json::value(arr) << ' '
                        << args.out_shape << " <- " << in_shape;
    throw std::runtime_error("pad not implemented");
}

static dlibx::net::layer_ptr
make_layer_reshape(const layer_args& args) {
    using T = dlibx::transpose_;
    assert(args.in_shapes.size() == 1);
    auto& in_shape = args.in_shapes.front();
    if (args.params.size() == 1) {
        auto& p = *args.params.front();
        assert(p.k() == 1 && p.nr() == 1 && p.nc() == 1);
        json::array arr(p.begin(), p.end());
        FILE_LOG(logINFO) << "reshape: " << json::value(arr)
                          << ' ' << args.out_shape << " <- " << in_shape;
        if (args.out_shape != in_shape) {
            FILE_LOG(logWARNING) << "reshape: "
                                 << in_shape << " -> " << args.out_shape;
            assert(shape_size(args.out_shape) == shape_size(in_shape));
        }
        assert(p.num_samples() <= 3);
        long k =  1 <= p.num_samples() ? std::lround(p.host()[0]) : 1;
        long nr = 2 <= p.num_samples() ? std::lround(p.host()[1]) : 1;
        long nc = 3 <= p.num_samples() ? std::lround(p.host()[2]) : 1;
        return std::make_unique<
            dlibx::net::layer_generic<T> >(dlibx::TRANSPOSE_KRC, k, nr, nc);
    }
    assert(args.params.size() == 0);
    FILE_LOG(logINFO) << "reshape: " << args.out_shape << " <- " << in_shape;
    assert(shape_size(args.out_shape) == shape_size(in_shape));
    return std::make_unique<
        dlibx::net::layer_generic<T> >(dlibx::TRANSPOSE_KRC, -1, 1, 1);
    //throw std::runtime_error("reshape not implemented");
}

static dlibx::net::layer_ptr
make_layer_prelu(const layer_args& args) {
    assert(args.in_shapes.size() == 1 && args.params.size() == 1);
    auto& in_shape = args.in_shapes.front();
    assert(args.out_shape == in_shape);
    auto& p = *args.params.front();
    assert(p.num_samples() == 1 && p.k() == 1 && p.nc() == 1);
    //FILE_LOG(logINFO) << "prelu: " << p.nr();
    struct sub {
        long n;
        inline auto& get_output() const { return *this; }
        inline long k() const { return n; }
    };
    using T = dlibx::prelu_;
    T prelu(0.25, true);
    prelu.setup(sub{p.nr()});
    auto& dt = prelu.get_layer_params();
    std::copy_n(p.host(), p.nr(), dt.host_write_only());
    return std::make_unique<dlibx::net::layer_generic<T> >(prelu);
}

static dlibx::net::layer_ptr
make_layer_relu(const layer_args& args) {
    assert(args.in_shapes.size() == 1 && args.params.size() == 0);
    auto& in_shape = args.in_shapes.front();
    assert(args.out_shape == in_shape);
    return std::make_unique<dlibx::net::layer_generic<dlib::relu_> >();
}

static dlibx::net::layer_ptr
make_layer_logistic(const layer_args& args) {
    assert(args.in_shapes.size() == 1 && args.params.size() == 0);
    auto& in_shape = args.in_shapes.front();
    assert(args.out_shape == in_shape);
    return std::make_unique<dlibx::net::layer_generic<dlib::sig_> >();
}

static dlibx::net::layer_ptr
make_layer_add(const layer_args& args) {
    if (args.params.size() == 0) {
        assert(2 <= args.in_shapes.size());
        for (auto& in_shape : args.in_shapes)
            assert(args.out_shape == in_shape);
        return std::make_unique<dlibx::net::layer_add_prev>();
    }
    FILE_LOG(logERROR) << "unsupported add: "
                       << args.in_shapes.size() << " inputs + "
                       << args.params.size() << " params";
    throw std::runtime_error("not implemented");
}

static dlibx::net::layer_ptr
make_layer_concat(const layer_args& args) {
    assert(2 <= args.in_shapes.size() && args.params.size() == 0);
    unsigned k = 0;
    for (auto& in_shape : args.in_shapes) {
        assert(args.out_shape[0] == in_shape[0]);
        assert(args.out_shape[2] == in_shape[2]);
        assert(args.out_shape[3] == in_shape[3]);
        k += in_shape[1];
    }
    assert(args.out_shape[1] == k);
    switch (args.in_shapes.size()) {
    case 2:
        FILE_LOG(logINFO) << "concat: " << args.out_shape << " <- "
                          << args.in_shapes[0] << ' ' << args.in_shapes[1];
        return std::make_unique<dlibx::net::layer_concat<2> >();

    default:
        FILE_LOG(logERROR) << "concat: " << args.in_shapes.size();
    }
    throw std::runtime_error("concat not implemented");
}

dlibx::net::layer_ptr
conv::make_layer(const tflite::OperatorCode& opcode,
                 const layer_args& args) {
    const auto code = builtin_code(opcode);
    switch (code) {
    case tflite::BuiltinOperator_ADD:
        return make_layer_add(args);
    case tflite::BuiltinOperator_CONV_2D:
        return make_layer_conv2d(args);
    case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        return make_layer_dconv2d(args);
    case tflite::BuiltinOperator_LOGISTIC:
        return make_layer_logistic(args);
    case tflite::BuiltinOperator_MAX_POOL_2D:
        return make_layer_maxpool2d(args);
    case tflite::BuiltinOperator_RESHAPE:
        return make_layer_reshape(args);
    case tflite::BuiltinOperator_PAD:
        return make_layer_pad(args);
    case tflite::BuiltinOperator_RELU:
        return make_layer_relu(args);
    case tflite::BuiltinOperator_PRELU:
        return make_layer_prelu(args);
    case tflite::BuiltinOperator_CONCATENATION:
        return make_layer_concat(args);
    default:
        FILE_LOG(logERROR) << "unsupported operator: ["
                           << unsigned(code) << "] "
                           << tflite::EnumNamesBuiltinOperator()[code];
        throw std::runtime_error("unsupported operator");
    }
}

template <typename T>
inline T& deref(T* ptr) {
    assert(ptr);
    return *ptr;
}

template <typename T>
inline auto& single_element(T* ptr) {
    assert(ptr && ptr->size() == 1);
    return deref((*ptr)[0]);
}

tflite_model::tflite_model(const std::filesystem::path& path)
    : fbmodel_ptr(tflite::FlatBufferModel::BuildFromFile(path.c_str())),
      fbmodel(deref(fbmodel_ptr.get())),
      model(deref(fbmodel.GetModel())),
      buffers(deref(model.buffers())),
      opcodes(deref(model.operator_codes())),
      subgraph(single_element(model.subgraphs())),
      sg_tensors(deref(subgraph.tensors())),
      sg_operators(deref(subgraph.operators())) {

    assert(fbmodel.initialized());
    assert(buffers.size() > 0 && opcodes.size() > 0);
    assert(sg_tensors.size() > 0 && sg_operators.size() > 0);

    const auto check_offsetvector = [](auto& vec) {
        for (auto* ptr : vec)
            assert(ptr);
    };
    check_offsetvector(buffers);
    check_offsetvector(opcodes);
    check_offsetvector(sg_tensors);
    check_offsetvector(sg_operators);

    auto& sg_inputs = deref(subgraph.inputs());
    assert(sg_inputs.size() == 1 && sg_inputs[0] == 0);
    auto& t0 = deref(sg_tensors[0]);
    assert(t0.type() == tflite::TensorType_FLOAT32);
    assert(!t0.is_variable());
    assert(t0.buffer() < buffers.size());
    auto& b0 = deref(buffers[t0.buffer()]);
    assert(!b0.data());
    input_shape = to_shape(deref(t0.shape()));
    assert(input_shape[0] == 1 &&
           0 < input_shape[1] &&
           0 < input_shape[2] &&
           input_shape[3] == 3);
    
    auto& sg_outputs = deref(subgraph.outputs());
    assert(0 < sg_outputs.size());
    for (auto idx : sg_outputs) {
        assert(0 <= idx && unsigned(idx) < sg_tensors.size());
        auto& t = deref(sg_tensors[unsigned(idx)]);
        assert(t.type() == tflite::TensorType_FLOAT32);
        assert(!t.is_variable());
        assert(t.buffer() < buffers.size());
        auto& b = deref(buffers[t.buffer()]);
        assert(!b.data());
        output_tensor_index.push_back(unsigned(idx));
    }

    /* Each tensor is one of:
     *   variable  -> does not contain data
     *   !variable -> does not contain data (net input or output)
     *   !variable -> does contains data (model parameters)
     */
    for (auto* ptr : sg_tensors) {
        auto& t = deref(ptr);
        assert(t.buffer() < buffers.size());
        if (t.is_variable()) {
            auto& b = deref(buffers[t.buffer()]);
            assert(!b.data());
        }
    }

    tensors.resize(sg_tensors.size());
}

void tflite_model::log_metadata(applog::log_level level) const {
    static const auto clean_string = [](auto s) {
        for (auto c : s)
            if (c < ' ' || 127 <= c) {
                //return json::encode_json(s);
                return "(" + std::to_string(s.size()) + " bytes) " +
                    json::make_string(stdx::binary(s));
            }
        return s;
    };

    FILE_LOG(level) << "minimum runtime: "
                      << fbmodel.GetMinimumRuntime();

    for (auto& pr : fbmodel.ReadAllMetadata()) {
        auto s = pr.second;
        while (!s.empty() && s.back() == 0)
            s.pop_back();
        FILE_LOG(level) << '\t' << pr.first
                          << '\t' << clean_string(s);
    }
    assert(fbmodel.CheckModelIdentifier());

    auto& model = *fbmodel.GetModel();
    FILE_LOG(level) << "version: " << model.version();
    if (auto ptr = model.description())
        FILE_LOG(level) << "description: " << ptr->str();
    if (auto ptr = model.metadata_buffer())
        FILE_LOG(level) << "metadata_buffer: " << ptr->size();
    if (auto ptr = model.metadata())
        FILE_LOG(level) << "metadata: " << ptr->size();
    if (auto ptr = model.signature_defs())
        FILE_LOG(level) << "signature_defs: " << ptr->size();
    if (auto ptr = model.buffers())
        FILE_LOG(level) << "buffers: " << ptr->size();
    if (auto ptr = model.operator_codes()) {
        FILE_LOG(logNONE) << "operator_codes: " << ptr->size();
        json::array opcodes_arr;
        for (auto* ptr : opcodes)
            opcodes_arr.push_back(builtin_code(deref(ptr)));
        std::sort(opcodes_arr.begin(), opcodes_arr.end());
        FILE_LOG(level) << "operator codes: " << json::value(opcodes_arr);
    }
    if (auto ptr = model.subgraphs()) {
        FILE_LOG(level) << "subgraphs: " << ptr->size();
        FILE_LOG(level) << "subgraph name: " << deref(subgraph.name()).c_str();
    }
    FILE_LOG(level) << "subgraph tensors: " << sg_tensors.size();
    FILE_LOG(level) << "subgraph operators: " << sg_operators.size();
    FILE_LOG(level) << "input tensor: " << input_shape;
}

unsigned tflite_model::copy_float32_and_int32_params() {
    unsigned count = 0;
    for (unsigned idx = 0; idx < sg_tensors.size(); ++idx) {
        tflite::Tensor const& src = deref(sg_tensors[unsigned(idx)]);
        if (src.is_variable() ||
            (src.type() != tflite::TensorType_FLOAT32 &&
             src.type() != tflite::TensorType_INT32))
            continue;

        assert(src.buffer() < buffers.size());
        tflite::Buffer const& buffer = deref(buffers[src.buffer()]);
        auto* data_ptr = buffer.data();
        if (!data_ptr) continue;
        auto const& data = *data_ptr;

        auto* src_shape_ptr = src.shape();
        assert(src_shape_ptr);
        const auto shape = to_shape(*src_shape_ptr);
        auto size = shape_size(shape);

        assert(data.size() == 4*size);

        auto& dest_tensor = tensors[idx];
        assert(dest_tensor.size() == 0);
        dest_tensor.set_size(shape[0],shape[1],shape[2],shape[3]);
        float* dp = dest_tensor.host_write_only();

        switch (src.type()) {
        case tflite::TensorType_INT32: {
            auto* sp = reinterpret_cast<int32_t const*>(data.data());
            for ( ; size > 0; --size, ++sp, ++dp) {
                *dp = float(*sp);
                assert(*sp == std::lround(*dp));
            }
            break;
        }

        case tflite::TensorType_FLOAT32: {
            auto* sp = reinterpret_cast<float const*>(data.data());
            std::copy_n(sp, size, dp);
            break;
        }

        default:
            throw std::runtime_error("not implemented");
        }
        ++count;
    }
    return count;
}

unsigned tflite_model::dequantize_params() {
    unsigned count = 0;
    for (auto* ptr : sg_operators) {
        tflite::Operator const& op = deref(ptr);
        assert(op.opcode_index() < opcodes.size());
        tflite::OperatorCode const& opcode = deref(opcodes[op.opcode_index()]);
        if (builtin_code(opcode) != tflite::BuiltinOperator_DEQUANTIZE)
            continue;

        auto& outputs = deref(op.outputs());
        assert(outputs.size() == 1);
        const auto out_idx = outputs[0];
        assert(0 <= out_idx && unsigned(out_idx) < sg_tensors.size());
        auto& out_tensor = tensors[unsigned(out_idx)];
        assert(out_tensor.size() == 0);

        tflite::Tensor const& dest = deref(sg_tensors[unsigned(out_idx)]);
        assert(dest.type() == tflite::TensorType_FLOAT32);
        assert(dest.is_variable() == false);
        assert(dest.buffer() == 0);

        auto& inputs = deref(op.inputs());
        assert(inputs.size() == 1);
        const auto in_idx = inputs[0];
        assert(0 <= in_idx && unsigned(in_idx) < sg_tensors.size());
        assert(tensors[unsigned(in_idx)].size() == 0);

        tflite::Tensor const& src = deref(sg_tensors[unsigned(in_idx)]);
        assert(src.type() != tflite::TensorType_FLOAT32);
        assert(src.is_variable() == false);
        assert(0 < src.buffer() && src.buffer() < buffers.size());

        tflite::Buffer const& buffer = deref(buffers[src.buffer()]);
        auto const& data = deref(buffer.data());

        const auto dest_shape = to_shape(deref(dest.shape()));
        const auto src_shape = to_shape(deref(src.shape()));
        assert(src_shape == dest_shape);
        const auto size = shape_size(src_shape);

        const auto bytes_per_el = unsigned(data.size() / size);
        assert(data.size() == size * bytes_per_el);

        dequantize(data, src.type(), bytes_per_el,
                   dest_shape, out_tensor);
        ++count;
    }   
    return count;
}

