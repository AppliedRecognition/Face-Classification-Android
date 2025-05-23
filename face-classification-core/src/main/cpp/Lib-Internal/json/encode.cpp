
#include "encode.hpp"
#include "visit.hpp"

#include <stdext/base64.hpp>

#include <applog/core.hpp>

#include <cctype>
#include <iomanip>
#include <stdexcept>


using namespace json;


void json::detail::encode_string(std::string& out, const std::string_view& sv) {
    for (auto c : sv) {
        if (0 <= c && c < 32) {
            switch (c) {
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                std::stringstream o;
                o << "\\u" << std::hex << std::setfill('0') << std::setw(4)
                  << (unsigned(c)&0xffu);
                out += o.str();
            }
        }
        else if (c == '"')
            out += "\\\"";
        else if (c == '\\')
            out += "\\\\";
        else
            out += c;
    }
}

void json::encode(std::string& out, std::string_view sv) {
    out += '"';
    detail::encode_string(out,sv);
    out += '"';
}
std::ostream& json::encode(std::ostream& out, std::string_view sv,
                           const string& /*indent*/) {
    const auto max_len = detail::manip_max_string >= 0 ? 
        std::size_t(out.iword(detail::manip_max_string)) : 0u;
    std::string result;
    if (max_len > 0 && sv.size() > max_len) {
        encode(result,sv.substr(0,max_len));
        result += "++";
    }
    else
        encode(result,sv);
    return out << result;
}

// base64 encode
void json::encode(std::string& out, const binary& b) {
    out += '"';
    auto src = b.data<unsigned char>();
    auto len = b.size();
    char buf[4];
    while (len >= 3) {
        stdx::base64_encode3(buf,src,3);
        out.append(buf,4);
        src += 3;
        len -= 3;
    }
    if (len > 0) {
        stdx::base64_encode3(buf,src,len);
        out.append(buf,4);
    }
    out += '"';
}
std::ostream& json::encode(std::ostream& out, const binary& b, 
                           const string& /*indent*/) {
    if (detail::manip_binary_subst >= 0 && 
        out.pword(detail::manip_binary_subst)) {
        const auto str = std::string_view(
            static_cast<char*>(out.pword(detail::manip_binary_subst)));
        const auto n = str.find("###");
        if (n < str.size())
            return out << str.substr(0,n) << b.size() << str.substr(n+3);
        else
            return out << str;
    }
    std::string result;
    encode(result,b);
    return out << result;
}

template <>
void json::encode<value>(std::string& out, const value& v) {
    visit([&](const auto& x){encode(out,x);}, v);
}

template <>
std::ostream& json::encode<value>(
    std::ostream& out, const value& v, const string& indent) {
    visit([&](const auto& x){encode(out,x,indent);}, v);
    return out;
}

