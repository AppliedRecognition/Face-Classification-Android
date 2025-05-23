#pragma once

#include "net_layer_impl_input.hpp"
#include "net_layer_impl_inplace.hpp"
#include "net_layer_impl_fc.hpp"
#include "net_layer_impl_con.hpp"
#include "net_layer_impl_pool.hpp"
#include "net_layer_impl_tags.hpp"
 
#include "dnn_padding.hpp"
#include "dnn_resize.hpp"
#include "dnn_transpose.hpp"
#include "dnn_extract.hpp"

namespace dlibx {
    namespace net {

        // **** padding

        template <long top, long bottom, long left, long right>
        auto layer_code(const dlibx::padding_<top,bottom,left,right>&) {
            std::stringstream ss;
            ss << "padding_" << top;
            if (bottom != top || left != top || right != top) {
                ss << '_' << bottom;
                if (left != top || right != bottom)
                    ss << '_' << left << '_' << right;
            }
            return ss.str();
        }
        template <long top, long bottom, long left, long right>
        constexpr auto layer_type(const dlibx::padding_<top,bottom,left,right>&) {
            return "padding";
        }
        template <long top, long bottom, long left, long right>
        inline auto layer_concise(const dlibx::padding_<top,bottom,left,right>&) {
            std::stringstream ss;
            ss << "pad" << top;
            if (bottom != top)
                ss << '+' << bottom;
            if (left != top || right != bottom) {
                ss << 'x' << left;
                if (right != left)
                    ss << '+' << right;
            }
            return ss.str();
        }
        template <long top, long bottom, long left, long right>
        auto layer_json(const dlibx::padding_<top,bottom,left,right>&) {
            json::object config;
            config["padding"] = json::array {
                json::array{top,bottom}, json::array{left,right}
            };
            config["data_format"] = "channels_last";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "ZeroPadding2D";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** extract

        template <long OFS, long K, long NR, long NC>
        inline auto layer_code(const dlib::extract_<OFS,K,NR,NC>&) {
            std::stringstream ss;
            ss << "extract_" << K;
            if (NR > 1 || NC > 1 || OFS > 0) {
                ss << '_' << NR << '_' << NC;
                if (OFS > 0)
                    ss << '_' << OFS;
            }
            return ss.str();
        }
        template <long OFS, long K, long NR, long NC>
        constexpr auto layer_type(const dlib::extract_<OFS,K,NR,NC>&) {
            return "extract";
        }
        template <long OFS, long K, long NR, long NC>
        inline auto layer_concise(const dlib::extract_<OFS,K,NR,NC>&) {
            std::stringstream ss;
            ss << "extract" << K;
            if (NR > 1 || NC > 1 || OFS > 0) {
                ss << 'x' << NR << 'x' << NC;
                if (OFS > 0)
                    ss << '+' << OFS;
            }
            return ss.str();
        }
        template <long OFS, long K, long NR, long NC>
        inline auto layer_json(const dlib::extract_<OFS,K,NR,NC>&) {
            json::object config;
            config["output_shape"] = { NR, NC, K };
            config["offset"] = OFS;
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Reshape";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }

        constexpr auto layer_code(const dlibx::extract_&) {
            return "extract";
        }
        constexpr auto layer_type(const dlibx::extract_&) {
            return "extract";
        }
        inline auto layer_concise(const dlibx::extract_& ex) {
            std::stringstream ss;
            ss << "extract" << ex.k();
            if (ex.nr() > 1 || ex.nc() > 1)
                ss << 'x' << ex.nr() << 'x' << ex.nc();
            if (ex.offset() > 0)
                ss << '+' << ex.offset();
            return ss.str();
        }
        inline auto layer_json(const dlibx::extract_& ex) {
            json::object config;
            config["output_shape"] = { ex.nr(), ex.nc(), ex.k() };
            config["offset"] = ex.offset();
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Reshape";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** upsample

        template <int sy, int sx>
        inline auto layer_code(const dlib::upsample_<sy,sx>&) {
            auto s = "upsample_" + std::to_string(sy);
            if (sx != sy) {
                s += '_';
                s += std::to_string(sx);
            }
            return s;
        }
        template <int sy, int sx>
        constexpr auto layer_type(const dlib::upsample_<sy,sx>&) {
            return "upsample";
        }
        template <int sy, int sx>
        inline auto layer_concise(const dlib::upsample_<sy,sx>&) {
            auto s = "up" + std::to_string(sy);
            if (sx != sy) {
                s += 'x';
                s += std::to_string(sx);
            }
            return s;
        }
        template <int sy, int sx>
        inline auto layer_json(const dlib::upsample_<sy,sx>&) {
            json::object config;
            config["scale"] = { sy, sx };
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Resize";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** resize

        constexpr auto layer_code(const dlibx::resize_&) {
            return "resize";
        }
        constexpr auto layer_type(const dlibx::resize_&) {
            return "resize";
        }
        inline auto layer_concise(const dlibx::resize_& r) {
            std::stringstream ss;
            ss << "resize";
            if (r.nr() > 0 && r.nc() > 0) {
                ss << r.nr();
                if (r.nr() != r.nc())
                    ss << 'x' << r.nc();
            }
            return ss.str();
        }
        inline auto layer_json(const dlibx::resize_& r) {
            json::object config;
            if (r.nr() > 0 && r.nc() > 0)
                config["output_size"] = { r.nr(), r.nc() };
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Resize";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** transpose

        constexpr auto layer_code(const dlibx::transpose_&) {
            return "transpose";
        }
        constexpr auto layer_type(const dlibx::transpose_&) {
            return "transpose";
        }
        inline auto layer_concise(const dlibx::transpose_& r) {
            std::stringstream ss;
            ss << "t" << to_string(r.mode());
            if (r.k() || r.nr() || r.nc()) {
                char c = 0;
                for (auto x : { r.k(), r.nr(), r.nc() }) {
                    if (c) ss << c;
                    if (x > 0) ss << x;
                    else ss << (x < 0 ? '%' : '#');
                    c = 'x';
                }
            }
            return ss.str();
        }
        inline auto layer_json(const dlibx::transpose_& r) {
            json::object config;
            config["input"] = { r.k(), r.nr(), r.nc() };
            config["output"] = to_string(r.mode());
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Transpose";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }



        /** \brief General regular (not in place) layer.
         *
         * For layers providing the forward() method.
         * Only layers that do not access any previously tagged inputs
         * may be handled by this class.
         * For layers that do access previously tagged inputs,
         * a specialization is required.
         */
        template <typename DETAILS>
        struct layer_regular final : public layer {
            DETAILS detail;
            template <typename... Args>
            layer_regular(Args&&... args)
                : detail(std::forward<Args>(args)...) {}

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_regular>(detail);
            }
            
            void forward_const(dlib::tensor const* const* inputs,
                               std::size_t num_inputs) override {
                if (num_inputs != 1 || !inputs || !*inputs)
                    throw std::invalid_argument("single input expected");
                auto& out = allocate_output();
                detail.forward(tagged_input<0>(inputs), out);
            }

            dlib::tensor& get_layer_params() override {
                return detail.get_layer_params();
            }
            const dlib::tensor& get_layer_params() const override {
                return detail.get_layer_params();
            }

            description layer_description() const override {
                using ulong = unsigned long;
                return { layer_type(detail),
                         layer_concise(detail),
                         ulong(layer_output_size(detail)),
                         ulong(layer_parameter_count(detail)) };
            }
            std::string code() const override {
                return layer_code(detail);
            }
            json::array keras_array() const override {
                return layer_json(detail);
            }
            void serialize_detail(std::ostream& out) const override {
                serialize(detail, out);
            }
        };


        /** \brief Select between inplace and regular layers.
         */
        template <typename DETAILS, typename = void>
        struct is_inplace : std::false_type {};
        template <typename DETAILS>
        struct is_inplace<DETAILS,std::void_t<decltype(std::declval<DETAILS&>().forward_inplace(std::declval<const dlib::tensor&>(),std::declval<dlib::tensor&>()))> >
            : std::true_type {};

        template <typename DETAILS>
        using layer_generic = std::conditional_t<
            is_inplace<DETAILS>::value,
            layer_inplace<DETAILS>,
            layer_regular<DETAILS> >;


        /// useful typedefs
        using layer_relu = layer_generic<dlib::relu_>;
        using layer_multiply = layer_generic<dlib::multiply_>;
        using layer_affine = layer_generic<dlib::affine_>;
        using layer_bncon = layer_generic<dlib::bn_<dlib::CONV_MODE> >;
        using layer_bnfc = layer_generic<dlib::bn_<dlib::FC_MODE> >;


        // default implementations
        template <typename T>
        constexpr auto layer_type(const T& item) {
            return layer_code(item);
        }
        template <typename T>
        constexpr auto layer_concise(const T& item) {
            return layer_type(item);
        }
    }
}
