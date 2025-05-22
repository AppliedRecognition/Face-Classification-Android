
#include "types.hpp"
#include "visit.hpp"
#include "encode.hpp"
#include "pull_encode_amf3.hpp"
#include "pull_encode_cbor.hpp"
#include "push_decode_amf3.hpp"
#include "push_decode_json.hpp"

#include <stdext/base64.hpp>

#include <applog/core.hpp>

#include <cmath>
#include <cstdlib>


using namespace json;


bad_get::bad_get(const char* expected, const char* found)
    : msg(std::string("json::bad_get: expected ")
          + expected + " but found " + found) {
}

const value& object::operator[](const string& key) const {
    static const value null_value;
    const auto it = find(key);
    return it != end() ? it->second : null_value;
}

template <>
int value::compare(const value& v) const noexcept {
    return visit(
        [this](auto const& a) noexcept { return this->compare(a); }, v);
}

const char* json::type_name(const value& v) {
    return visit(type_name_visitor{}, v);
}

namespace {
    struct make_boolean_visitor {
        inline boolean operator()(null_type) const noexcept {
            return false;
        }
        inline boolean operator()(boolean b) const noexcept {
            return b;
        }
        inline boolean operator()(integer i) const noexcept {
            return i;
        }
        inline boolean operator()(real r) const noexcept {
            return std::fpclassify(r) != FP_ZERO;
        }
        inline boolean operator()(const string& s) const noexcept {
            return !s.empty() && s != "0";
        }
        inline boolean operator()(const binary& b) const noexcept {
            return !b.empty();
        }
        inline boolean operator()(const array& a) const noexcept {
            return !a.empty();
        }
        inline boolean operator()(const object& o) const noexcept {
            return !o.empty();
        }
    };
}

boolean json::make_boolean(const value& v) {
    return visit(make_boolean_visitor{},v);
}

integer json::make_integer(const value& v) {
    if (is_type<string>(v)) {
        auto& s = get<string>(v);
        return std::atol(s.c_str());
    }
    return get<integer>(v);
}

real json::make_real(const value& v) {
    if (is_type<string>(v)) {
        auto& s = get<string>(v);
        return std::atof(s.c_str());
    }
    return is_type<integer>(v) ? real(get<integer>(v)) : get<real>(v);
}

string json::make_string(const value& v) {
    return is_type<binary>(v) ?
        stdx::base64_encode(get<binary>(v)) : get<string>(v);
}

binary json::make_binary(const value& v) {
    return is_type<string>(v) ?
        stdx::base64_decode(get<string>(v)) : get<binary>(v);
}

void json::encode_json(std::string& out, const value& v) {
    encode(out,v);
}

static value raw_decode_json(std::string in_str) {
    // todo: why does decoder_input_type need a non-const string ?
    decoder_input_type in_pair(in_str);
    value_pusher out;
    decoder_input_fn in_fn = push_decode_json([&](auto v) { out = v; });
    in_fn(in_pair);
    if (!in_pair.data) {
        FILE_LOG(logDETAIL) << "decode_json: sending end-of-stream";
        in_fn(in_pair);  // decoder must finish or throw exception
    }
    return move_value(out);
}
value json::decode_json(const char* data, std::size_t size) {
    return raw_decode_json({data,size});
}
value json::decode_json(const std::string& in) {
    return raw_decode_json(in);
}

binary json::encode_amf3(const value& v) {
    return pull_encode_amf3(v,8192,512).pull_final();
}

value json::decode_amf3(const void* data, std::size_t size) {
    std::string in_str(static_cast<const char*>(data),size);
    decoder_input_type in_pair(in_str);
    value_pusher out;
    decoder_input_fn in_fn = push_decode_amf3([&](auto v) { out = v; });
    in_fn(in_pair);
    if (!in_pair.data) {
        FILE_LOG(logDETAIL) << "decode_amf3: sending end-of-stream";
        in_fn(in_pair);  // decoder must finish or throw exception
    }
    return move_value(out);
}
value json::decode_amf3(const binary& in) {
    return decode_amf3(in.data(),in.size());
}

binary json::encode_cbor(const value& v) {
    return pull_encode_cbor(v,8192,512).pull_final();
}

value json::decode_any(const void* data, std::size_t size) {
    if (size > 0) {
        const auto header = *static_cast<const unsigned char*>(data);
        if (header < 32)
            return decode_amf3(data,size);
        if (128 <= header)
            return decode_cbor(data,size);
    }
    return decode_json(static_cast<const char*>(data), size);
}
value json::decode_any(const binary& in) {
    if (!in.empty()) {
        const auto header = *in.data<unsigned char>();
        if (header < 32)
            return decode_amf3(in);
        if (128 <= header)
            return decode_cbor(in);
    }
    return decode_json(in.data<char>(), in.size());
}

std::istream& json::operator>>(std::istream& in, array& out) {
    json::value v;
    in >> v;
    out = get_array(v);
    return in;
}
std::istream& json::operator>>(std::istream& in, object& out) {
    json::value v;
    in >> v;
    out = get_object(v);
    return in;
}
std::istream& json::operator>>(std::istream& in, value& out) {
    value_pusher result;
    auto infn = push_decode_json([&](value_pusher p){result=p;});
    std::string buf;
    for (;;) {
        buf.clear();
        for (;;) {
            const auto c = in.get();
            if (c == std::istream::traits_type::eof())
                break;
            buf.push_back(char(c));
            if (c == '"' || c == ']' || c == '}' || isspace(c))
                break;
        }
        if (buf.empty()) {
            decoder_input_type din;
            infn(din);
            break;
        }
        decoder_input_type din(buf);
        infn(din);
        if (din.data) {
            if (din.pos != buf.end()) {
                assert(next(din.pos) == buf.end());
                in.unget();
            }
            break;
        }
    }
    out = move_value(result);
    return in;
}

std::ostream& json::operator<<(std::ostream& out, const json::array& a) {
    return encode(out,a);
}
std::ostream& json::operator<<(std::ostream& out, const json::object& o) {
    return format_object(out,o.begin(),o.end());
}
std::ostream& json::operator<<(std::ostream& out, const value& v) {
    return encode(out,v);
}
