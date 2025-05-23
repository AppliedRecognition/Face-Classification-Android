
#include "ncnn_common.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <applog/core.hpp>

void ncnn::load_model(core::context_data& data,
                      models::type model_type,
                      std::string_view model_name,
                      Net& net) {
    const auto& loader = det::internal::get_loader(data);
    auto&& r = loader(models::format::ncnn, model_type, model_name);
    auto& vec = r.models;
    int failed = 2;
    if (vec.size() >= 2) {
        if (!r.path.empty())
            FILE_LOG(logINFO) << "face detector: " << r.path;
        // .param file
        if (auto p = std::get_if<models::istream_ptr>(&vec[0])) {
            if (auto s = p->get()) {
                // have to read entire param file into stdx::binary
                // because istream_reader::scan() is not implemented
                std::vector<char> buf(
                    std::istreambuf_iterator<char>(*s), {});
                vec[0] = stdx::binary(move(buf));
            }
        }
        if (auto p = std::get_if<stdx::binary>(&vec[0])) {
            if (!p->empty()) {
                auto buf = p->data<unsigned char>();
                ncnn::DataReaderFromMemory in(buf);
                if (auto e = net.load_param(in))
                    FILE_LOG(logERROR)
                        << "ncnn::Net::load_param() error " << e;
                else --failed;
            }
        }
        // .bin file
        if (auto p = std::get_if<models::istream_ptr>(&vec[1])) {
            if (auto s = p->get()) {
                ncnn::istream_reader in(*s);
                if (auto e = net.load_model(in))
                    FILE_LOG(logERROR)
                        << "ncnn::Net::load_model() error "
                        << e;
                else --failed;
            }
        }
        else if (auto p = std::get_if<stdx::binary>(&vec[1])) {
            if (!p->empty()) {
                auto buf = p->data<unsigned char>();
                ncnn::DataReaderFromMemory in(buf);
                if (auto e = net.load_model(in))
                    FILE_LOG(logERROR)
                        << "ncnn::Net::load_model() error " << e;
                else --failed;
            }
        }
    }
    if (failed)
        throw std::runtime_error("failed to load ncnn model");
}
