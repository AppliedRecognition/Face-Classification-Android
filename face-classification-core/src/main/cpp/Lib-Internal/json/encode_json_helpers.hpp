#pragma once

#include "types.hpp"
#include "encode.hpp"
#include <stdext/base64.hpp>

namespace {
    inline bool check_for_json_chars(std::string_view str) {
        for (auto c : str)
            if (unsigned(c) < 32 || c == '\\' || c == '"')
                return true;
        return false;
    }

    json::string encode_string(const json::string& p) {
        if (!check_for_json_chars(p))
            return p;  // string doesn't need encoding
        
        /* Note that we don't escape the forward slash '/' character because:
         *  + the json spec at json.org seems to indicate that it can be
         *    escaped but does not have to be
         *  + base64 uses the / character so there is a big performance penalty
         *    to encoding a string just because of the presence of the slash
         */
        
        std::string out;
        out.reserve(11*p.length()/10+1);  // assume 10% expansion
        json::detail::encode_string(out,p);
        return out;
    }

    json::string encode_binary(const json::binary& p, 
                               std::basic_string<unsigned char>& pre_input) {
        auto input = p.data<unsigned char>();
        auto input_len = p.size();
            
        std::string out;
        out.reserve(4*(input_len+pre_input.size())/3 + 4);  // 4/3 expansion
            
        if (!pre_input.empty()) {
            while (pre_input.length() < 3 && input_len > 0) {
                pre_input += *input++;
                --input_len;
            }
            if (pre_input.length() == 3) {
                char dest[4];
                stdx::base64_encode3(dest,pre_input.data(),pre_input.length());
                out.append(dest,4);
                pre_input.clear();
            }
        }
            
        while (input_len >= 3) {
            char dest[4];
            stdx::base64_encode3(dest,input,3);
            out.append(dest,4);
            input += 3;
            input_len -= 3;
        }
            
        if (input_len > 0)
            pre_input.append(input,input_len);
        
        return out;
    }

    json::string finish_binary(std::basic_string<unsigned char>& pre_input) {
        // finish base64 encode
        std::string out;
        if (!pre_input.empty()) {
            char dest[4];
            stdx::base64_encode3(dest,pre_input.data(),pre_input.length());
            out.append(dest,4);
            pre_input.clear();
        }
        return out;
    }
}
