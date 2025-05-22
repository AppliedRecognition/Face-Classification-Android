
#ifndef __LIB_INTERNAL_BUILD_APPLICATIONS__
#error nv-inspect.cpp is an application -- not part of the dlibx library
#endif

#include "net_vector.hpp"
#include "net_layer.hpp"
#include "library_init.hpp"
#include "bfloat16.hpp"

#include <raw_image/input_extractor.hpp>

#include <json/types.hpp>
#include <json/io_manip.hpp>

#include <applog/applog.hpp>

#include <filesystem>
#include <iostream>


using namespace dlibx;

static auto output_type_and_size(const net::vector& model) {
    json::object o;
    auto ts = model.output_type_and_size();
    if (!ts.first.empty())
        o["type"] = ts.first;
    if (ts.second)
        o["size"] = ts.second;
    return o;
}

static auto parameter_count(const net::vector& model) {
    decltype(net::layer::description::parameters) n = 0;
    for (auto& layer : model)
        n += layer.layer_description().parameters;
    return n;
}

static auto parameter_encoding(const net::vector& model) {
    std::map<parameter_format, unsigned> distr;
    for (auto& layer : model)
        ++distr[layer.parameter_format()];
    json::object o;
    for (auto& p : distr) {
        if (p.first >= pf::quantize_base) {
            std::stringstream ss;
            ss << std::setfill('0') << bits_per_element(p.first) << 'q';
            o[ss.str()] = p.second;
        }
        else switch (p.first) {
            case pf::bfloat16: o["16f"] = p.second; break;
            case pf::float32:  o["32f"] = p.second; break;
            case pf::native:   o["none"] = p.second; break;
            default: o["unknown"] = p.second;
            }
    }
    return o;
}

static auto round_bits(double x) {
    return std::round(100*x) / 100;
}

int main(int argc, char*argv[]) {
    const char* const prog = [&](){
        assert(argc > 0);
        if (auto p = strrchr(argv[0], '/')) return p + 1;
        return argv[0];
    }();
    ++argv, --argc;
    (void)prog;

    if (argc <= 0) {
        FILE_LOG(logFATAL) << "usage:" << std::endl
                           << "\tnv-inspect model_file";
        return 1;
    }
    const auto model_filename = std::filesystem::path(*argv);
    ++argv, --argc;
    if (!is_regular_file(model_filename)) {
        FILE_LOG(logFATAL) << "file not found: " << model_filename;
        return 1;
    }

    // library init
    library_init();

    // load model
    net::vector model(std::ifstream(model_filename,std::ios::binary));
    assert(!model.empty());

    const auto num_params = parameter_count(model);
    const auto model_size = file_size(model_filename);
    const auto bits_per_param = 8 * double(model_size) / double(num_params);

    // layers
    auto layers = json::object {
        { "count",      model.size() },
        { "parameters", json::object {
                { "count",    num_params },
                { "mean_bits", round_bits(bits_per_param) },
                { "encoding", parameter_encoding(model) }
            } },
        { "structure", json::object {
                { "generic", model.concise() }
            } }
    };
    
    // input
    json::value input;
    if (auto e = model.input_extractor) {
        input = json::object {
            { "name",   e->name },
            { "width",  e->width },
            { "height", e->height },
            { "pixel",  to_string(e->layout) }
        };
        try {
            auto copy = model;
            auto img = create(e->width, e->height, e->layout);
            std::vector<float> out;
            copy(img, out);
            get_object(layers["structure"])["detail"] = copy.concise();
        }
        catch (const std::exception& e) {
            FILE_LOG(logWARNING) << e.what();
            get_object(layers["structure"])["detail"] = e.what();
        }
    }

    // final description
    auto top = json::object {
        { "filename", model_filename.filename().generic_string() },
        { "filesize", model_size },
        { "layers",   move(layers) },
        { "input",    move(input) },
        { "output",   output_type_and_size(model) },
        { "labels",   model.labels },
        { "meta",     model.meta }
    };

    std::cout << json::indent("  ") << top << std::endl;

    return 0;
}
