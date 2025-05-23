
#include <datareader.h>

#include <raw_image/input_extractor.hpp>

#include "models.hpp"

#include <core/context.hpp>
#include <core/thread_data.hpp>

#include <applog/core.hpp>

#include <stdext/binarystream.hpp>


namespace ncnn {
    class istream_reader : public DataReader {
    public:
        std::istream& in;
        istream_reader(std::istream& in) : in(in) {}
        size_t read(void* buf, size_t size) const override {
            in.read(static_cast<char*>(buf),std::streamsize(size));
            return size_t(in.gcount());
        }
        //virtual int scan(const char* format, void* p) const;
  };
}

std::shared_ptr<rec::ncnn::model_record>
rec::ncnn::load_shared(version_type ver, const core::context_data& cd) {
    const auto lptr = core::cptr<models_loader>(cd.context);
    if (!lptr) {
        FILE_LOG(logWARNING) << "models basepath not set for rec_ncnn";
        return nullptr;
    }
    auto mp = std::make_shared<model_record>();
    for (auto& pr : known_models) {
        if (pr.first == ver) {
            mp->extractor = raw_image::input_extractor::find(pr.second);
            break;
        }
    }
    if (!mp->extractor) {
        FILE_LOG(logWARNING) << "unknown ncnn model version: " << ver;
        return nullptr;
    }
    auto&& r = lptr->loader(models::format::ncnn,
                            models::type::face_recognition,
                            models::face_recognition(ver));
    auto& vec = r.models;
    if (vec.size() < 2) {
        FILE_LOG(logWARNING) << "failed to find ncnn recognition model: " << ver;
        return nullptr;
    }
    auto& net = mp->net;
    int failed = 2;
    try {
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
                ::ncnn::DataReaderFromMemory in(buf);
                if (auto e = net.load_param(in))
                    FILE_LOG(logERROR)
                        << "ncnn::Net::load_param() error " << e;
                else --failed;
            }
        }
        // .bin file
        if (auto p = std::get_if<models::istream_ptr>(&vec[1])) {
            if (auto s = p->get()) {
                ::ncnn::istream_reader in(*s);
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
                ::ncnn::DataReaderFromMemory in(buf);
                if (auto e = net.load_model(in))
                    FILE_LOG(logERROR)
                        << "ncnn::Net::load_model() error " << e;
                else --failed;
            }
        }
    }
    catch (const std::exception& e) {
        FILE_LOG(logERROR) << "failed to load recognition model: "
                           << e.what();
    }
    if (!failed) {
        FILE_LOG(logINFO) << "load[" << ver << "]: "
                          << (r.path.empty() ?
                              "(ncnn recognition model)" : r.path);
        return mp;
    }
    return nullptr;
}
