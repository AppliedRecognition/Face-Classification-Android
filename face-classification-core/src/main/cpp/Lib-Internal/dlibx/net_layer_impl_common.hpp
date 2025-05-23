#pragma once

#include "net_layer.hpp"
#include <dlib/dnn/input.h>

namespace dlibx {
    namespace net {
        template <typename T>
        constexpr auto layer_type(const T&);
        template <typename T>
        constexpr auto layer_concise(const T&);
        template <typename T>
        constexpr auto layer_output_size(const T&) { return 0u; }

        template <typename DETAILS, typename = void>
        struct layer_parameter_count {
            const DETAILS& layer;
            layer_parameter_count(const DETAILS& layer) : layer(layer) {}
            inline operator std::size_t() const {
                return layer.get_layer_params().size();
            }
        };
        template <typename DETAILS>
        struct layer_parameter_count<DETAILS, std::void_t<decltype(std::declval<DETAILS&>().get_num_params())> > {
            const DETAILS& layer;
            layer_parameter_count(const DETAILS& layer) : layer(layer) {}
            inline operator std::size_t() const {
                return layer.get_num_params();
            }
        };
        

        template <typename>
        struct layer_regular;

        /** \brief Input subnet with tagged tensors.
         *
         * The subnet tagged_input<COUNT> provides access to COUNT+1 tensors
         * having tag ids ranging from 0 to COUNT, inclusive.
         * Directly calling get_output() on the subnet gives the same tensor
         * as associated with tag id COUNT.
         * Note that the array of tensor pointers provided to the constructor
         * or the init() method are in reverse order.  
         * They are for tag ids COUNT, COUNT-1, ... 0.
         */
        template <std::size_t IDX, typename SUBNET>
        struct tagged_input_;
        
        template <typename SUBNET>
        struct tagged_input_<0,SUBNET> {
            static constexpr unsigned long id = 0;
            using input_type = typename SUBNET::input_type;
            using subnet_type = SUBNET;
            subnet_type s;
            const dlib::tensor* t;
            inline void init(dlib::tensor const* const* ts) { t = *ts; }
            tagged_input_() = default;
            tagged_input_(dlib::tensor const* const* ts) { init(ts); }
            auto& get_output() const { return *t; }
        };

        template <std::size_t IDX, typename SUBNET>
        struct tagged_input_ {
            static constexpr unsigned long id = IDX;
            using input_type = typename SUBNET::input_type;
            using subnet_type = SUBNET;
            subnet_type s;
            const dlib::tensor* t;
            inline void init(dlib::tensor const* const* ts) {
                t = *ts;
                s.init(ts + 1);
            }
            tagged_input_() = default;
            tagged_input_(dlib::tensor const* const* ts) { init(ts); }
            auto& get_output() const { return *t; }
            inline auto& subnet() { return s; }
            inline auto& subnet() const { return s; }
        };

        template <std::size_t COUNT>
        struct tagged_input_select;
        template <>
        struct tagged_input_select<0> {
            struct empty_subnet {
                using input_type = dlib::input<dlib::matrix<float> >;
            };
            using type = tagged_input_<0,empty_subnet>;
        };
        template <std::size_t COUNT>
        struct tagged_input_select {
            using type = tagged_input_<COUNT, typename tagged_input_select<COUNT-1>::type>;
        };

        template <std::size_t COUNT>
        using tagged_input = typename tagged_input_select<COUNT>::type;

        template <typename SUBNET>
        using input_tag_0 = tagged_input_<0,SUBNET>;
    }
}
