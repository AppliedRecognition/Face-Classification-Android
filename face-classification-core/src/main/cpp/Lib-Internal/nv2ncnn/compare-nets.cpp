
#include <dlibx/net_vector.hpp>
#include <dlibx/net_layer.hpp>
#include <dlibx/net_layer_impl.hpp>
#include <dlibx/input_extractor.hpp>
#include <dlibx/library_init.hpp>

#include <raw_image/io.hpp>
#include <raw_image/ncnn.hpp>

#include <applog/core.hpp>

// ncnn
#include <mat.h>
#include <net.h>
#include <datareader.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>



// **** main ****

int main(int argc, char*argv[]) {
    const char* const prog = [&](){
        assert(argc > 0);
        if (auto p = strrchr(argv[0], '/')) return p + 1;
        return argv[0];
    }();
    ++argv, --argc;

    if (argc < 2) {
        FILE_LOG(logFATAL) << "Usage:" << std::endl
                           << '\t' << prog << " model_file.nv image_file";
        return 1;
    }

    const auto nv_path = std::filesystem::path(*argv);
    ++argv, --argc;
    if (!is_regular_file(nv_path)) {
        FILE_LOG(logFATAL) << "file not found: " << nv_path;
        return 1;
    }

    const auto img_path = std::filesystem::path(*argv);
    ++argv, --argc;
    if (!is_regular_file(img_path)) {
        FILE_LOG(logFATAL) << "file not found: " << img_path;
        return 1;
    }

    const auto param_path = nv_path.filename().replace_extension(".param");
    const auto bin_path =   nv_path.filename().replace_extension(".bin");
    if (!exists(param_path) || !exists(bin_path)) {
        FILE_LOG(logERROR) << "ncnn model not found:" << std::endl
                           << '\t' << param_path << " and " << std::endl
                           << '\t' << bin_path;
        return 1;
    }

    // library init
    dlibx::library_init();

    // load nv model
    auto nv_model =
        dlibx::net::vector(std::ifstream(nv_path,std::ios::binary));
    assert(!nv_model.empty());

    // load ncnn model
    ncnn::Net ncnn_model;
    {
        auto pin = fopen(param_path.c_str(), "r");
        ncnn::DataReaderFromStdio pread(pin);
        ncnn_model.load_param(pread);
        fclose(pin);

        auto bin = fopen(bin_path.c_str(), "rb");
        ncnn::DataReaderFromStdio bread(bin);
        ncnn_model.load_model(bread);
        fclose(bin);
    }
    assert(ncnn_model.input_names().size() == 1);
    const auto ncnn_input = ncnn_model.input_names().front();

    // load image
    const auto img = load(img_path, raw_image::pixel::rgb24);
    FILE_LOG(logINFO) << diag(img);
    
    // nv model inference
    std::vector<dlib::resizable_tensor> nv_out(8);
    if (auto n = nv_model(img, nv_out)) {
        assert(n <= nv_out.size());
        nv_out.resize(n);
        FILE_LOG(logINFO) << "nv output tensors: " << n;
        for (dlib::tensor const& t : nv_out) {
            auto* data = t.host();
            std::stringstream ss;
            for (unsigned i = 0; i < std::min(4u, unsigned(t.size())); ++i)
                ss << ' ' << data[i];
            FILE_LOG(logINFO) << "-> " << t.size() << ss.str();
        }
    }
    else throw std::runtime_error("no output from nv model");

    // ncnn model inference
    // extractor setup
    auto ex = ncnn_model.create_extractor();
    ex.input(ncnn_input, to_ncnn_rgb(img));

    // output
    FILE_LOG(logINFO) << "ncnn outputs: " << ncnn_model.output_names().size();
    for (auto name : ncnn_model.output_names()) {
        ncnn::Mat blob;
        ex.extract(name, blob);
        assert(blob.total() % 4 == 0);
        const auto size = unsigned(blob.total() / 4);
        std::stringstream ss;
        for (unsigned i = 0; i < std::min(4u, size); ++i)
            ss << ' ' << blob[i];
        FILE_LOG(logINFO) << "-> " << size << ss.str();
    }

    {
        // dlib outputs
        std::ofstream nv_diag("diag-nv.txt");
        nv_diag << "-- nv --" << std::endl;
        std::string prev_line;
        unsigned prev_size = 0;
        for (auto& layer : nv_model) {
            std::stringstream line;
            auto& t = layer.last_output();
            line << t.size();
            if (prev_size != t.size()) {
                prev_size = unsigned(t.size());
                nv_diag << std::endl;
            }
            auto data = t.host();
            for (unsigned i = 0; i < std::min(16u, unsigned(t.size())); ++i)
                line << ' ' << data[i];
            auto str = line.str();
            if (prev_line != str)
                nv_diag << (prev_line = move(str)) << std::endl;
        }
    }

    {
        // ncnn outputs
        std::ofstream ncnn_diag("diag-ncnn.txt");
        ncnn_diag << "-- ncnn --" << std::endl;
        std::string prev_line;
        unsigned prev_size = 0;
        for (int i = 0; i < 1000; ++i) {
            auto ex = ncnn_model.create_extractor();
            ex.input(ncnn_input, to_ncnn_rgb(img));
            ncnn::Mat data;
            if (ex.extract(i, data) != 0)
                continue;

            std::stringstream line;
            //assert(data.total() % 4 == 0);
            //const auto size = unsigned(data.total() / 4);
            const auto size = unsigned(data.total());
            line << size;
            if (prev_size != size) {
                prev_size = size;
                ncnn_diag << std::endl;
            }

            assert(data.dims == 3 && data.d == 1);
            //ncnn_diag << ' ' << data.w << 'x' << data.h << 'x' << data.c;
            if (size != data.w*data.h*data.c)
                line << " ***";
        
            for (unsigned i = 0; i < std::min(16u, size); ++i)
                line << ' ' << data[i];

            auto str = line.str();
            if (prev_line != str)
                ncnn_diag << (prev_line = move(str)) << std::endl;
        }
    }
    
}
