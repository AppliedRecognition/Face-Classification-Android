
#include "types.hpp"

#include <stdext/bswap.hpp>

using namespace json;

namespace {
    struct decoder {
        uint8_t const* data;
        std::size_t size;

        value decode();

        uint64_t integer() {
            if (size <= 0)
                throw std::runtime_error(
                    "insufficient bytes for cbor integer");
            const auto arg = unsigned(*data) & 0x1f;
            if (arg < 24) {
                ++data, --size;
                return arg;
            }
            unsigned nbytes;
            switch (arg) {
            case 24: nbytes = 1; break;
            case 25: nbytes = 2; break;
            case 26: nbytes = 4; break;
            case 27: nbytes = 8; break;
            default:
                throw std::runtime_error("invalid cbor integer variant");
            }
            if (size < 1 + nbytes)
                throw std::runtime_error(
                    "insufficient bytes for cbor integer");
            uint64_t x = 0;
            for (unsigned i = 1; i <= nbytes; ++i)
                x = (x<<8) + data[i];
            data += 1 + nbytes;
            size -= 1 + nbytes;
            return x;
        }

        value special() {
            if (size <= 0)
                throw std::runtime_error(
                    "insufficient bytes for cbor floating point");
            const auto arg = unsigned(*data) & 0x1f;
            if (arg < 24) {
                ++data, --size;
                switch (arg) {
                case 20: return false;
                case 21: return true;
                case 22: return null;
                case 23: return null; // "undefined" but we return null
                }
                throw std::runtime_error("unknown cbor simple value");
            }
            switch (arg) {
            case 24: {
                if (size < 2)
                    throw std::runtime_error(
                        "insufficient bytes for cbor simple value");
                data += 2, size -= 2;
                throw std::runtime_error("unknown cbor simple value");
            }
            case 25: {
                if (size < 3)
                    throw std::runtime_error(
                        "insufficient bytes for cbor floating point");
                data += 3, size -= 3;
                throw std::runtime_error("cbor float16 not implemented");
            }
            case 26: {
                if (size < 5)
                    throw std::runtime_error(
                        "insufficient bytes for cbor floating point");
                float f;
                stdx::copy_be(data+1, data+5, reinterpret_cast<uint8_t*>(&f));
                data += 5, size -= 5;
                return f;
            }
            case 27: {
                if (size < 9)
                    throw std::runtime_error(
                        "insufficient bytes for cbor floating point");
                double f;
                stdx::copy_be(data+1, data+9, reinterpret_cast<uint8_t*>(&f));
                data += 9, size -= 9;
                return f;
            }
            }
            throw std::runtime_error("invalid cbor float variant");
        }

        std::string string_or_binary() {
            if (size <= 0)
                throw std::runtime_error(
                    "insufficient bytes for cbor string or binary");
            const auto header = *data;
            const auto type = header >> 5;
            if (type != 2 && type != 3)
                throw std::runtime_error("expected cbor string or binary");
            std::string str;
            if ((header&0x1f) != 0x1f) {
                const auto len = integer();
                if (size < len)
                    throw std::runtime_error(
                        "insufficient bytes for cbor string or binary");
                str.assign(reinterpret_cast<const char*>(data),len);
                data += len, size -= len;
            }
            else { // indefinite length
                ++data, --size;
                for (;;) {
                    if (size <= 0)
                        throw std::runtime_error("insufficient bytes for cbor indefinite string or binary");
                    const auto code = *data;
                    if (code == 0xff) {
                        ++data, --size;
                        break;
                    }
                    if ((code>>5) != type)
                        throw std::runtime_error(
                            "invalid cbor string or binary chunk");
                    const auto len = integer();
                    if (size < len)
                        throw std::runtime_error(
                            "insufficient bytes for cbor string");
                    str.append(reinterpret_cast<const char*>(data),len);
                    data += len, size -= len;
                }
            }
            return str;
        }

        json::array array() {
            if (size <= 0)
                throw std::runtime_error(
                    "insufficient bytes for cbor array");
            const auto header = *data;
            if ((header >> 5) != 4)
                throw std::runtime_error("expected cbor array");
            json::array arr;
            if ((header&0x1f) != 0x1f) {
                auto len = integer();
                arr.reserve(len);
                for ( ; 0 < len; --len)
                    arr.emplace_back(decode());
            }
            else { // indefinite length
                ++data, --size;
                for (;;) {
                    if (size <= 0)
                        throw std::runtime_error(
                            "insufficient bytes for cbor indefinite array");
                    if (*data == 0xff) {
                        ++data, --size;
                        break;
                    }
                    arr.emplace_back(decode());
                }
            }
            return arr;
        }

        object map() {
            if (size <= 0)
                throw std::runtime_error(
                    "insufficient bytes for cbor map");
            const auto header = *data;
            if ((header >> 5) != 5)
                throw std::runtime_error("expected cbor map");
            object obj;
            if ((header&0x1f) != 0x1f) {
                for (auto len = integer(); 0 < len; --len) {
                    auto key = decode();
                    obj.emplace(move(get_string(key)), decode());
                }
            }
            else { // indefinite length
                ++data, --size;
                for (;;) {
                    if (size <= 0)
                        throw std::runtime_error(
                            "insufficient bytes for cbor indefinite map");
                    if (*data == 0xff) {
                        ++data, --size;
                        break;
                    }
                    auto key = decode();
                    obj.emplace(move(get_string(key)), decode());
                }
            }
            return obj;
        }
    };
}

value decoder::decode() {
    if (size <= 0)
        throw std::runtime_error("insufficient bytes for cbor value");
    switch (*data >> 5) {
    case 0: return integer();
    case 1: return -1 - json::integer(integer());
    case 2: return binary(string_or_binary());
    case 3: return string_or_binary(); // string
    case 4: return array();
    case 5: return map();
    case 6: // tagged data item
        integer(); // ignore the tag (could handle this in the future)
        return decode();
    case 7: return special();
    }
    assert(!"machine failure");
    throw std::runtime_error("machine failure");
}

value json::decode_cbor(const void* data_, std::size_t size) {
    if (!data_ || !size)
        throw std::invalid_argument(
            "empty data buffer passed to json::decode_cbor");
    return decoder{static_cast<uint8_t const*>(data_),size}.decode();
}

value json::decode_cbor(const binary& in) {
    return decode_cbor(in.data(), in.size());
}
