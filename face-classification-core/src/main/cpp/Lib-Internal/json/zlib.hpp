#pragma once

#include <functional>

#include "pull_types.hpp"
#include "push_types.hpp"

namespace json {

    /** \brief From first 2 bytes test if data is deflate compressed.
     *
     * Ascii first byte that might pass test: (8HXhx
     * No json or amf3 will pass the test.
     */
    inline bool is_compressed(unsigned char byte0, unsigned char byte1) {
        return (byte0&0x0f) == 8 && (byte0*256u+byte1)%31 == 0;
    }
    inline bool is_compressed(const void* data) {
        const auto p = static_cast<const unsigned char*>(data);
        return is_compressed(p[0],p[1]);
    }

    binary inflate(const void* data, std::size_t size);

    
    binary_puller pull_deflate(const binary_puller& input,
                               unsigned buffer_size = 1024);
    inline binary_puller pull_deflate(const string_puller& input,
                                      unsigned buffer_size = 1024) {
        return pull_deflate(pull_binary(input,convert_cast),buffer_size);
    }

    string_puller pull_inflate_string(const binary_puller& input,
                                      unsigned buffer_size = 1024);
    inline binary_puller pull_inflate_binary(const binary_puller& input,
                                             unsigned buffer_size = 1024) {
        return pull_binary(pull_inflate_string(input,buffer_size),convert_cast);
    }


    using binary_push_function =
        std::function<void(std::optional<json::binary>)>;
    binary_push_function push_deflate_fn(binary_push_function output,
                                         unsigned buffer_size = 1024,
                                         bool sync = false);

    binary_pusher push_deflate(const binary_pusher& output,
                               unsigned buffer_size = 1024);
    
    binary_pusher push_inflate(const string_pusher& output,
                               unsigned buffer_size = 1024);
    inline binary_pusher push_inflate(const binary_pusher& output,
                                      unsigned buffer_size = 1024) {
        return push_inflate(get_string_pusher(output,convert_cast),buffer_size);
    }
}
