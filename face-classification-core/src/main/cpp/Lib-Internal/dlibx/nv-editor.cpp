
#ifndef __LIB_INTERNAL_BUILD_APPLICATIONS__
#error nv-editor.cpp is an application -- not part of the dlibx library
#endif

#include "net_vector.hpp"
#include "library_init.hpp"
#include "bfloat16.hpp"

#include <json/types.hpp>
#include <json/io_manip.hpp>

#include <applog/applog.hpp>

#include <filesystem>


using namespace dlibx;


int main(int argc, char*argv[]) {
    const char* const prog = [&](){
        assert(argc > 0);
        if (auto p = strrchr(argv[0], '/')) return p + 1;
        return argv[0];
    }();
    ++argv, --argc;
    (void)prog;

    if (argc <= 0) {
        FILE_LOG(logFATAL) << "usage:" << std::endl << "\tnv-editor model_file"
            " [ labels=[\"classA\",...] ]"
            " [ meta=json ] [ meta_key=json ]"
            " [ --format=f32|bf16|q16|...|q4 ]";
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

    // model details before changes
    FILE_LOG(logINFO) << "--";
    FILE_LOG(logINFO) << "filename: " << model_filename.filename();
    FILE_LOG(logINFO) << "description: "
                      << json::indent("\t") << model.description();
    FILE_LOG(logINFO) << "--";

    if (argc <= 0)
        return 0;  // nothing more to do

    const auto output_size = model.output_type_and_size().second;

    auto format = pf::native;
    std::string_view format_str;
    for ( ; argc > 0; ++argv, --argc) {
        const auto sv = std::string_view(*argv);
        const auto eq = sv.find('=');
        if (0 < eq && eq < sv.size()) {
            const auto key = sv.substr(0,eq);
            if (key == "--format") {
                format_str = sv.substr(eq+1);
                if (format_str == "f32")
                    format = pf::float32;
                else if (format_str == "bf16")
                    format = pf::bfloat16;
                else if (format_str.front() == 'q' && format_str.size() > 1)
                    format = quantize(atol(format_str.data()+1));
                else {
                    FILE_LOG(logERROR) << "unknown format: " << format_str;
                    return 1;
                }
                continue;
            }
            auto val = json::decode_json(sv.data()+eq+1,sv.size()-eq-1);
            if (key == "label" || key == "labels") {
                if (json::is_type<json::string>(val))
                    model.labels = { get_string(val) };
                else if (json::is_type<json::array>(val)) {
                    const auto& arr = get_array(val);
                    if (arr.size() > 1 && arr.size() != output_size) {
                        FILE_LOG(logERROR) << "labels must match outputs: "
                                           << val;
                        throw std::invalid_argument("bad labels");
                    }
                    std::vector<std::string> new_labels;
                    for (auto& v : get_array(val))
                        new_labels.push_back(get_string(v));
                    model.labels = move(new_labels);
                }
                else {
                    FILE_LOG(logERROR) << "bad labels: " << val;
                    throw std::invalid_argument("bad labels");
                }
                FILE_LOG(logINFO) << "new labels: "
                                  << json::value(model.labels);
            }
            else if (key == "meta") {
                model.meta = move(get_object(val));
                FILE_LOG(logINFO) << "new meta: "
                                  << json::indent("\t") << model.meta;
            }
            else if (key.size() > 5 && key.compare(0,5,"meta_") == 0) {
                const auto k = std::string(key.substr(5));
                auto& v = model.meta[k] = val;
                FILE_LOG(logINFO) << "new meta[\"" << k << "\"]: "
                                  << json::indent("\t") << v;
            }
            else {
                FILE_LOG(logERROR) << "unknown setting: " << key;
                return 1;
            }
        }
        else {
            FILE_LOG(logERROR) << "invalid arg: " << sv;
            return 1;
        }
    }

    bool found = false;
    auto fn = model_filename;
    if (!format_str.empty()) {
        fn.replace_extension(".nv-" + std::string(format_str));
        if (!exists(fn))
            found = true;
    }
    if (!found) {
        for (auto ext =
                 std::string(".nv-new0"); ext.back() <= '9'; ++ext.back()) {
            fn.replace_extension(ext);
            if (!exists(fn)) {
                found = true;
                break;
            }
        }
    }
    if (found) {
        FILE_LOG(logINFO) << "writing -> " << fn;
        dlib::serialize(fn.native()) << format << model;
    }
    else
        FILE_LOG(logWARNING) << "NOT OVERWRITING EXISTING FILES!";

    FILE_LOG(logINFO) << "--";
    
    return 0;
}
