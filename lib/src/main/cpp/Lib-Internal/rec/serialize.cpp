
#include "serialize.hpp"
#include "internal_serialize.hpp"
#include <json/zlib.hpp>
#include <stdext/convert.hpp>

using namespace rec;

bool internal::is_compressed(const void* src, std::size_t len) {
    return len >= 2 && json::is_compressed(src);
}
stdx::binary internal::remove_compression(const void* src, std::size_t len) {
    return json::inflate(src, len);
}

bool internal::is_prototype(const void* data, std::size_t size) {
    if (size < 4) return false;
    // note: amf3 object is either: 0A 0B 01 or 0A 01
    auto p = static_cast<const uint8_t*>(data);
    if (16 < p[0] && p[0] < 120 && p[1] == 1 && size == 132)
        return true; // special short 128 x 8-bit format
    return p[0] != 0 && p[2] != 0 && (p[2]&0xec) == 0 &&
        p[3] != 0 && p[3] <= 2;
}
        
std::vector<std::pair<const void*, std::size_t> >
internal::deserialize_multiple(const void* data, std::size_t size) {
    if (size < 12)
        throw std::runtime_error(
            "invalid multi-prototype serialization (too short)");

    std::vector<std::pair<const void*, std::size_t> > result;

    const uint32_t* p = static_cast<const uint32_t*>(data);
    const auto header = deserialize_value<uint32_t>(p++);
    if (header & 0xff)
        result.emplace_back(data, size); // single prototype

    else {
        if (((header^(1<<24))&0xffff00ff) != 0)
            throw std::runtime_error(
                "invalid multi-prototype serialization (format)");
        const auto version = (header>>8) & 0xff;
        size -= 4;
        
        while (const auto len = deserialize_value<uint32_t>(p++)) {
            const auto n = (len+3)/4;
            if (size < 8 + 4*n)
                throw std::runtime_error(
                    "invalid multi-prototype serialization (too short)");
            size -= 4 + 4*n;
            if ((deserialize_value<uint32_t>(p) & 0xff) != version)
                throw std::runtime_error(
                    "invalid multi-prototype serialization (version)");
            result.emplace_back(p,len);
            p += n;
        }
    }

    return result;
}

stdx::binary rec::to_binary_with_opts(
    const json::value& val,
    const stdx::options_tuple<serialize_type,compression_type>& opts) {

    stdx::binary result;
    switch (std::get<serialize_type>(opts)) {
    case serialize_type::json:
        result = stdx::binary(json::encode_json(val));
        break;

    case serialize_type::cbor:
        result = json::encode_cbor(val);
        break;

    case serialize_type::amf3:
    case serialize_type::def:
    case serialize_type::raw:  ///< invalid_argument?
        result = json::encode_amf3(val);
        break;
    }

    if (std::get<compression_type>(opts) != rec::uncompressed)
        result = json::pull_deflate(result).pull_final();
    return result;
}
