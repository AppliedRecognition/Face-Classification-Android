
#ifndef __LIB_INTERNAL_BUILD_APPLICATIONS__
#error nv-profile.cpp is an application -- not part of the dlibx library
#endif

#include "net_vector.hpp"
#include "library_init.hpp"
#include "bfloat16.hpp"

#include <raw_image/input_extractor.hpp>
#include <raw_image/transform.hpp>
#include <raw_image_io/io.hpp>

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <json/types.hpp>
#include <json/io_manip.hpp>

#include <stdext/convert.hpp>
#include <stdext/rounding.hpp>

#include <applog/applog.hpp>

#include <filesystem>
#include <chrono>
#include <random>

static std::mt19937 rgen(1);

template <typename T>
static auto median(const std::vector<T>& vec) {
    assert(!vec.empty());
    const auto h = vec.size() / 2;
    return (vec.size()&1) ? vec[h] : (vec[h-1]+vec[h])/2;
}

int main(int argc, char*argv[]) {
    const char* const prog = [&](){
        assert(argc > 0);
        if (auto p = strrchr(argv[0], '/')) return p + 1;
        return argv[0];
    }();
    ++argv, --argc;
    (void)prog;

    // settings
    core::context_settings cs;
    cs.max_threads = 64; // auto detect
    unsigned num_runs = 1;
    std::filesystem::path model_filename;
    std::filesystem::path sample_filename;

    for ( ; argc > 0; ++argv, --argc) {
        auto arg = *argv;
        if (arg[0] == '-' && arg[1] == 't' && '0' < arg[2] && arg[2] <= '9') {
            cs.min_threads = cs.max_threads = stdx::convert_from(atoi(arg+2));
            if (cs.max_threads > 64) {
                FILE_LOG(logFATAL) << "invalid num_threads: " << arg;
                return 1;
            }
        }
        else if (arg[0] == '-' && arg[1] == 'n' &&
                 '0' < arg[2] && arg[2] <= '9') {
            num_runs = stdx::convert_from(atoi(arg+2));
        }
        else if (model_filename.empty()) {
            model_filename = arg;
            if (!is_regular_file(model_filename)) {
                FILE_LOG(logFATAL) << "file not found: " << model_filename;
                return 1;
            }
        }
        else if (sample_filename.empty()) {
            sample_filename = arg;
            if (!is_regular_file(sample_filename)) {
                FILE_LOG(logFATAL) << "file not found: " << sample_filename;
                return 1;
            }
        }
        // else if (output_vector.empty())
    }

    if (model_filename.empty()) {
        FILE_LOG(logFATAL) << "usage:" << std::endl
                           << "\tnv-profile [ -t# ] [ -n# ]"
                           << " model_file [ input_sample ] [ output_vector ]";
        return 1;
    }

    // library init
    dlibx::library_init();

    // load model
    dlibx::net::vector model(std::ifstream(model_filename,std::ios::binary));
    assert(!model.empty());
    
    // model details
    FILE_LOG(logINFO) << "--";
    FILE_LOG(logINFO) << " model: " << model_filename.filename();
    FILE_LOG(logINFO) << "  size: " << file_size(model_filename);
    FILE_LOG(logINFO) << "layers: " << model.size();

    const auto ts = model.output_type_and_size();
    FILE_LOG(logINFO) << "output: " << ts.first << ':' << ts.second;
    
    if (auto e = model.input_extractor) {
        json::object input;
        input["name"] = e->name;
        input["width"] = e->width;
        input["height"] = e->height;
        input["pixel"] = to_string(e->layout);
        FILE_LOG(logINFO) << " input: " << json::indent("\t") << input;
    }
    else
        FILE_LOG(logWARNING) << " input: nullptr";
    
    FILE_LOG(logINFO) << "  meta: " << json::indent("\t") << model.meta;
    FILE_LOG(logINFO) << "labels: " << json::value(model.labels);
    FILE_LOG(logINFO) << "--";

    // sample
    raw_image::plane_ptr sample;
    if (!sample_filename.empty()) {
        sample = raw_image::load(sample_filename);
        FILE_LOG(logINFO) << "sample loaded: "
                          << sample->width << 'x' << sample->height;
        if (auto e = model.input_extractor) {
            if (e->width != sample->width || e->height != sample->height ||
                e->layout != sample->layout) {
                FILE_LOG(logINFO) << "converting sample to: "
                                  << e->width << 'x' << e->height << ' '
                                  << to_string(e->layout);
                sample =
                    copy_resize(sample, e->width, e->height, e->layout);
            }
        }
    }
    else if (auto e = model.input_extractor) {
        FILE_LOG(logINFO) << "random sample: "
                          << e->width << 'x' << e->height << ' '
                          << to_string(e->layout);
        sample = raw_image::create(e->width, e->height, e->layout);
        std::normal_distribution<float> nd;
        const auto size = sample->bytes_per_line*sample->height;
        for (auto px = sample->data, end = px + size; px != end; ++px)
            *px = stdx::round_from(128 + 32*nd(rgen));
    }
    else {
        FILE_LOG(logFATAL) << "cannot generate sample without input_extractor";
        return 1;
    }

    if (num_runs <= 0) return 0;

    // context
    auto context = core::context::construct(cs);
    FILE_LOG(logINFO) << "number of threads: " << context->num_threads();

    // do runs
    FILE_LOG(logINFO) << "number of runs: " << num_runs;
    const auto sample_orig = copy(sample);
    std::vector<unsigned> runtime_ms;
    runtime_ms.reserve(num_runs);
    std::vector<std::vector<float> > outputs;
    outputs.reserve(num_runs);
    for (unsigned n = 1; n <= num_runs; ++n) {
        if (n > 1) {
            // add noise to sample
            copy_pixels(sample_orig, sample);
            std::normal_distribution<float> nd;
            const auto size = sample->bytes_per_line*sample->height;
            for (auto px = sample->data, end = px + size; px != end; ++px)
                *px = stdx::round_from(*px + 4*nd(rgen));
        }
        const auto job =
            [&]() {
                using namespace std::chrono;
                outputs.emplace_back();
                const auto start = steady_clock::now();
                model(sample, outputs.back());
                const auto dt = steady_clock::now() - start;
                return duration_cast<milliseconds>(dt).count();
            };
        runtime_ms.push_back(stdx::convert_from(context->threads().run(job)));
    }

    // results
    if (runtime_ms.size() < 2)
        FILE_LOG(logINFO) << "time: " << runtime_ms.front() << " milliseconds";
    else {
        std::sort(runtime_ms.begin(), runtime_ms.end());
        FILE_LOG(logINFO) << "times: (min/med/max) "
                          << runtime_ms.front() << '/'
                          << median(runtime_ms) << '/'
                          << runtime_ms.back() << " milliseconds";
    }

    FILE_LOG(logINFO) << "--";
    
    return 0;
}
