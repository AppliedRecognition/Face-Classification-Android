#pragma once

#include <mat.h>
#include <net.h>
#include <datareader.h>

#include <raw_image/ncnn.hpp>

#include <models/types.hpp>

#include <istream>

namespace core {
    struct context_data;
}

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

    // load ncnn model using models::loader
    void load_model(core::context_data& data,
                    models::type model_type,
                    std::string_view model_name,
                    Net& dest);
}
