#pragma once

#include "push_types.hpp"

#include <functional>
#include <optional>

namespace json {
    namespace detail {
        struct exception_handler_base {
            virtual ~exception_handler_base() {}
            virtual bool operator()(std::exception&) = 0;
        };
        template <typename T>
        struct exception_hander_obj : public exception_handler_base {
            T handler;
            exception_hander_obj(T handler) : handler(handler) {}
            bool operator()(std::exception& e) {
                return handler(e);
            }
        };
    }

    /// data provided to decoder input function
    struct decoder_input_type {
        std::string* data = nullptr;
        std::string::iterator pos;
        decoder_input_type() = default;
        decoder_input_type(std::string& data)
            : data(&data), pos(data.begin()) {}
    };

    /// stream decoder input function
    using decoder_input_fn = std::function<void(decoder_input_type&)>;

    /// function called when decoded value is ready
    using decoder_output_fn = std::function<void(value_pusher)>;
}


