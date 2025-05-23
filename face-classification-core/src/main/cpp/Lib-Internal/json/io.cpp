
#include "io.hpp"
#include "push_decode_cbor.hpp"
#include "push_decode_amf3.hpp"
#include "push_decode_json.hpp"
#include "pull_encode_cbor.hpp"
#include "pull_encode_amf3.hpp"
#include "pull_encode_json.hpp"
#include "zlib.hpp"

#include <applog/core.hpp>

using namespace json;

value internal::load(stdx::file_ptr file, const std::string& path) {
    if (!file) {
        FILE_LOG(logERROR) << "file '" << path
                           << "' could not be opened for reading";
        throw std::runtime_error("json/amf3 file not found");
    }
    char header[2];
    if (fread(header, 2, 1, file.get()) != 1) {
        FILE_LOG(logERROR) << "error reading from file '" << path << "'";
        throw std::runtime_error("error while reading from json/amf3 file");
    }
    string_puller puller;
    if (is_compressed(header)) {
        binary_puller in;
        in.push_back(stdx::binary(header));
        in.set_handler([fp=file.get()] {
            auto buf = std::make_unique<char[]>(64*1024);
            const auto n = fread(buf.get(),1,64*1024,fp);
            std::optional<stdx::binary> r;
            if (0 < n) r.emplace(move(buf),n);
            return r;
        });
        puller = pull_inflate_string(in);
    }
    else {
        puller.push_back(std::string(header,2));
        puller.set_handler([fp=file.get()] {
            std::optional<std::string> r(std::in_place, 64*1024, char(0));
            const auto n = fread(r->data(),1,64*1024,fp);
            if (n <= 0) r.reset();
            else r->resize(n);
            return r;
        });
    }
    auto block = puller();
    if (!block || block->empty()) {
        FILE_LOG(logERROR) << "error reading from file '" << path << "'";
        throw std::runtime_error("error while reading from file");
    }
    value_pusher out;
    decoder_input_fn in_fn;
    if (uint8_t(block->front()) < 32)
        in_fn = push_decode_amf3([&](auto v) { out = v; });
    else if (uint8_t(block->front()) < 128)
        in_fn = push_decode_json([&](auto v) { out = v; });
    else // >= 128
        in_fn = push_decode_cbor([&](auto v) { out = v; });
    for (;;) {
        auto in = decoder_input_type(*block);
        in_fn(in);
        if (in.data)
            break; // decoder done (but there is more input data)
        block = puller();
        if (!block) {
            in_fn(in); // signal eof
            break;
        }
    }
    return move_value(out);
}

static bool ends_with(const std::string& path, std::string_view ext) {
    if (path.size() < ext.size()) return false;
    return path.compare(path.size() - ext.size(), ext.size(), ext) == 0;
}

void internal::save(const value& val, FILE* outfile, const std::string& path,
                    const stdx::options_tuple<cbor_option,amf3_option,json_option,deflate_option>& opts) {
    binary_puller puller;

    auto cbor = std::get<cbor_option>(opts);
    auto amf3 = std::get<amf3_option>(opts);
    auto json = std::get<json_option>(opts);
    auto gz = std::get<deflate_option>(opts);

    if (0 == cbor + amf3 + json + gz) {
        // attempt to determine type from path
        if (ends_with(path, ".gz")) {
            gz.b = true;
            if (ends_with(path, ".cbor.gz"))
                cbor.b = true;
            else if (ends_with(path, ".json.gz"))
                json.b = true;
            else if (ends_with(path, ".amf3.gz"))
                amf3.b = true;
        }
        else if (ends_with(path, ".cbor"))
            cbor.b = true;
        else if (ends_with(path, ".json"))
            json.b = true;
        else if (ends_with(path, ".amf3"))
            amf3.b = true;
    }

    if (1 < cbor + amf3 + json)
        throw std::invalid_argument(
            "only select one of cbor, amf3 or json for json::save()");
    const char* type;

    if (json) {
        type = "json";
        puller = pull_binary(pull_encode_json(val,64*1024,1024),convert_cast);
    }
    else if (amf3) {
        type = "amf3";
        puller = pull_encode_amf3(val,64*1024,1024);
    }
    else { // cbor (default)
        type = "cbor";
        puller = pull_encode_cbor(val,64*1024,1024);
    }

    if (gz)
        puller = pull_deflate(puller);
    while (auto bin = puller()) {
        if (fwrite(bin->data(), bin->size(), 1, outfile) != 1) {
            FILE_LOG(logERROR) << "error writing " << type
                               << " to file '" << path << "'";
            throw std::runtime_error(
                std::string("error while writing ") + type + " to file");
        }
    }
}
