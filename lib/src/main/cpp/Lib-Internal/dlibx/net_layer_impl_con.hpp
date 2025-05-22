#pragma once

#include "net_layer_impl_common.hpp"
#include "dnn_lmcon.hpp"
#include "dnn_condw.hpp"

namespace dlibx {
    namespace net {

        template <long NR, long NC, int SY, int SX,
                  int PY, int PX, int DY = 1, int DX = 1>
        auto layer_code_construct(const char* base) {
            if (NR == 0 && NC == 0)
                return std::string(base) + "_all";
            std::stringstream ss;
            ss << base << '_' << NR;
            if (DY > 1) ss << 'd' << DY;
            ss << "_" << NC;
            if (DX > 1) ss << 'd' << DX;
            if (NR > 1 || NC > 1 || SY > 1 || SX > 1) {
                ss << '_' << SY << '_' << SX;
                if (PY > 0 || PX > 0)
                    ss << '_' << PY << '_' << PX;
            }
            return ss.str();
        }
        template <long NR, long NC, int SY, int SX,
                  int PY, int PX, int DY = 1, int DX = 1>
        auto layer_concise_construct(std::string s) {
            static_assert((NR == 0 && NC == 0 &&
                           PY == 0 && PX == 0) ||
                          (NR > 0 && NC > 0 &&
                           SY >= 1 && SX >= 1 &&
                           PY >= 0 && PX >= 0));
            if (NR > 0) {
                s += std::to_string(NR);
                if (NR > 1 && DY > 1) {
                    s += 'd';
                    s += std::to_string(DY);
                }
                if (NC != NR || DY != DX) {
                    s += 'x';
                    s += std::to_string(NC);
                    if (NC > 1 && DX > 1) {
                        s += 'd';
                        s += std::to_string(DX);
                    }
                }
                if (SY > 1 || SX > 1) {
                    s += '/';
                    s += std::to_string(SY);
                    if (SX != SY) {
                        s += 'x';
                        s += std::to_string(SX);
                    }
                }
                if (PY > 0 || PX > 0) {
                    s += "|pad";
                    s += std::to_string(PY);
                    if (PX != PY) {
                        s += 'x';
                        s += std::to_string(PX);
                    }
                }
            }
            return s;
        }
        template <long NR, long NC, int SY, int SX,
                  int PY, int PX, int DY = 1, int DX = 1>
        auto layer_json_con(long K, bool bias) {
            json::array arr;
            json::object config;
            config["activation"] = "linear";
            config["trainable"] = true;
            config["filters"] = K;
            config["dilation_rate"] = json::array{DY,DX};
            if (NR > 0 && NC > 0) {
                static constexpr auto WR = 1 + (NR-1)*DY;
                static constexpr auto WC = 1 + (NC-1)*DX;
                config["kernel_size"] =  json::array{NR,NC};
                config["strides"] = json::array{SY,SX};
                if (NR == 1 && NC == 1)
                    config["padding"] = "same";  // no change in image size
                else if (PY == 0 && PX == 0)
                    config["padding"] = "valid"; // no padding
                else if (1 && SY == 1 && SX == 1 && PY == WR/2 && PX == WC/2)
                    config["padding"] = "same";
                else {  // separate padding object
                    json::object zconfig;
                    zconfig["padding"] =
                        json::array{json::array{PY,PY}, json::array{PX,PX}};
                    zconfig["data_format"] = "channels_last";
                    zconfig["trainable"] = true;
                    //zconfig["dtype"] = "float32";
                    json::object zobj;
                    zobj["class_name"] = "ZeroPadding2D";
                    zobj["config"] = move(zconfig);
                    zobj["name"] = "padding";
                    arr.push_back(move(zobj));
                    config["padding"] = "valid";  // no padding
                }
            }
            // else ??
            config["use_bias"] = bias;
            config["dtype"] = "float32";
            config["data_format"] = "channels_last";
            json::object obj;
            obj["class_name"] = "Conv2D";
            obj["config"] = move(config);
            arr.push_back(move(obj));
            return arr;
        }


        // **** dlib::con

        template <long K, long NR, long NC, int... Is>
        constexpr auto layer_has_bias(const dlib::con_<K,NR,NC,Is...>& con) {
            return !bias_is_disabled(con);
        }
        template <long K, long NR, long NC, int... Is>
        auto layer_add_bias(dlib::con_<K,NR,NC,Is...>& con) {
            if (bias_is_disabled(con)) {
                const auto& old_params = con.get_layer_params();
                const auto kin = long(old_params.size())
                    / (con.num_filters()*con.nr()*con.nc());
                const dlib::resizable_tensor input{1,kin,con.nr(),con.nc()};
                struct subnet {
                    dlib::tensor const& t;
                    inline dlib::tensor const& get_output() const {
                        return t;
                    }
                };
                const auto kout = unsigned(con.num_filters());
                dlib::con_<K,NR,NC,Is...> newcon{kout};
                newcon.setup(subnet{input});
                auto& new_params = newcon.get_layer_params();
                DLIB_CASSERT(0 < old_params.size() &&
                             old_params.size() + kout == new_params.size());
                memcpy(new_params.host(), old_params.host(),
                       old_params.size() * sizeof(float));
                con = std::move(newcon);
            }
            return true;
        }
        template <long K, long NR, long NC, int... Is>
        auto layer_nonzero_bias(const dlib::con_<K,NR,NC,Is...>& con) {
            if (layer_has_bias(con)) {
                const auto k = con.num_filters();
                auto& params = con.get_layer_params();
                if (0 < k && std::size_t(k) <= params.size()) {
                    static constexpr auto eq = std::equal_to<>{};
                    for (auto end = params.host() + params.size(),
                             p = end - k; p != end; ++p)
                        if (!eq(*p,0)) return true;
                }
            }
            return false;
        }
        template <long K, long NR, long NC, int... Is>
        constexpr std::array<int,2>
        layer_dilate(const dlib::con_<K,NR,NC,Is...>&) {
            return { 1, 1 };
        }
        template <long K, long NR, long NC, int... Is>
        inline auto layer_code(const dlib::con_<K,NR,NC,Is...>&) {
            return layer_code_construct<NR,NC,Is...>("con");
        }
        template <long K, long NR, long NC, int... Is>
        constexpr auto layer_type(const dlib::con_<K,NR,NC,Is...>&) {
            return "con";
        }
        template <long K, long NR, long NC, int... Is>
        inline auto layer_concise(const dlib::con_<K,NR,NC,Is...>& con) {
            return layer_concise_construct<NR,NC,Is...>(
                layer_nonzero_bias(con) ? "bias|con" : "con");
        }
        template <long K, long NR, long NC, int... Is>
        inline auto layer_output_size(const dlib::con_<K,NR,NC,Is...>& con) {
            return unsigned(con.num_filters());
        }
        template <long K, long NR, long NC, int... Is>
        inline auto layer_json(const dlib::con_<K,NR,NC,Is...>& con) {
            return layer_json_con<NR,NC,Is...>(
                con.num_filters(), layer_nonzero_bias(con));
        }
        template <long K, long NR, long NC, int... Is>
        constexpr auto serialize_format(const dlib::con_<K,NR,NC,Is...>&) {
            return pf::float32;
        }


        // **** lm_con

        template <long K, long NR, long NC, int... Is>
        constexpr auto layer_has_bias(const dlibx::lm_con_<K,NR,NC,Is...>& con) {
            return con.get_bias_mode() == HAS_BIAS;
        }
        template <long K, long NR, long NC, int... Is>
        constexpr auto layer_add_bias(dlibx::lm_con_<K,NR,NC,Is...>& con) {
            con.add_biases();
            return true;
        }
        template <long K, long NR, long NC, int... Is>
        auto layer_nonzero_bias(const dlibx::lm_con_<K,NR,NC,Is...>& con) {
            if (layer_has_bias(con)) {
                const auto k = con.num_filters();
                auto& params = con.get_layer_params();
                if (0 < k && std::size_t(k) <= params.size()) {
                    static constexpr auto eq = std::equal_to<>{};
                    for (auto end = params.host() + params.size(),
                             p = end - k; p != end; ++p)
                        if (!eq(*p,0)) return true;
                }
            }
            return false;
        }
        template <long K, long NR, long NC, int... Is>
        constexpr std::array<int,2>
        layer_dilate(const dlibx::lm_con_<K,NR,NC,Is...>& con) {
            return { con.dilate_x(), con.dilate_y() };
        }
        template <long K, long NR, long NC, int... Is>
        inline auto layer_code(const dlibx::lm_con_<K,NR,NC,Is...>&) {
            return layer_code_construct<NR,NC,Is...>("con");
        }
        template <long K, long NR, long NC, int... Is>
        constexpr auto layer_type(const dlibx::lm_con_<K,NR,NC,Is...>&) {
            return "con";
        }
        template <long K, long NR, long NC, int... Is>
        inline auto layer_concise(const dlibx::lm_con_<K,NR,NC,Is...>& con) {
            return layer_concise_construct<NR,NC,Is...>(
                layer_nonzero_bias(con) ? "bias|con" : "con");
        }
        template <long K, long NR, long NC, int... Is>
        inline auto layer_output_size(const dlibx::lm_con_<K,NR,NC,Is...>& con) {
            return unsigned(con.num_filters());
        }
        template <long K, long NR, long NC, int... Is>
        inline auto layer_json(const dlibx::lm_con_<K,NR,NC,Is...>& con) {
            return layer_json_con<NR,NC,Is...>(
                con.num_filters(), layer_nonzero_bias(con));
        }


        // **** condw

        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        inline auto layer_has_bias(const dlibx::condw_<BM,MULT,NR,NC,Is...>& con) {
            return con.get_bias_mode() == HAS_BIAS;
        }
        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        inline auto layer_add_bias(dlibx::condw_<BM,MULT,NR,NC,Is...>& con) {
            con.add_biases();
            return true;
        }
        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        constexpr std::array<int,2>
        layer_dilate(const dlibx::condw_<BM,MULT,NR,NC,Is...>&) {
            return { 1, 1 };
        }
        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        inline auto layer_code(const dlibx::condw_<BM,MULT,NR,NC,Is...>&) {
            return layer_code_construct<NR,NC,Is...>("cdw");
        }
        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        constexpr auto layer_type(const dlibx::condw_<BM,MULT,NR,NC,Is...>&) {
            return "condw";
        }
        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        inline auto layer_concise(const dlibx::condw_<BM,MULT,NR,NC,Is...>& con) {
            return layer_concise_construct<NR,NC,Is...>(
                con.get_bias_mode() == HAS_BIAS ? "bias|cdw" : "cdw");
        }
        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        inline auto layer_output_size(const dlibx::condw_<BM,MULT,NR,NC,Is...>& con) {
            return unsigned(con.num_filters());
        }
        template <bias_mode BM, long MULT, long NR, long NC,
                  int SY, int SX, int PY, int PX>
        auto layer_json(const dlibx::condw_<BM,MULT,NR,NC,SY,SX,PY,PX>& con) {
            json::array arr;
            json::object config;
            config["activation"] = "linear";
            config["trainable"] = true;
            config["depth_multiplier"] = MULT;
            config["dilation_rate"] = json::array(2,1);
            if (NR > 0 && NC > 0) {
                config["kernel_size"] =  json::array{NR,NC};
                config["strides"] = json::array{SY,SX};
                if (NR == 1 && NC == 1)
                    config["padding"] = "same";  // no change in image size
                else if (PY == 0 && PX == 0)
                    config["padding"] = "valid"; // no padding
                else if (1 && SY == 1 && SX == 1 && PY == NR/2 && PX == NC/2)
                    config["padding"] = "same";
                else {  // separate padding object
                    json::object zconfig;
                    zconfig["padding"] =
                        json::array{json::array{PY,PY}, json::array{PX,PX}};
                    zconfig["data_format"] = "channels_last";
                    zconfig["trainable"] = true;
                    //zconfig["dtype"] = "float32";
                    json::object zobj;
                    zobj["class_name"] = "ZeroPadding2D";
                    zobj["config"] = move(zconfig);
                    zobj["name"] = "padding";
                    arr.push_back(move(zobj));
                    config["padding"] = "valid";  // no padding
                }
            }
            // else ??
            config["use_bias"] = con.get_bias_mode() == HAS_BIAS;
            config["dtype"] = "float32";
            config["data_format"] = "channels_last";
            json::object obj;
            obj["class_name"] = "DepthwiseConv2D";
            obj["config"] = move(config);
            arr.push_back(move(obj));
            return arr;
        }


        /** \brief Specialization for con_ (to allow special operations).
         */
        struct layer_con : public layer {
            virtual bool has_bias() const = 0;
            virtual bool add_bias() = 0;
            virtual long num_filters() const = 0;
            virtual long nr() const = 0;
            virtual long nc() const = 0;
            virtual std::array<int,2> dilate() const = 0;
            virtual std::array<int,2> stride() const = 0;
            virtual std::array<int,2> padding() const = 0;
        };
        template <typename CON>
        struct layer_con_t : layer_con {
            CON detail;

            template <typename... Args>
            layer_con_t(Args&&... args)
                : detail(std::forward<Args>(args)...) {}

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_con_t>(detail);
            }

            dlib::tensor& get_layer_params() override {
                return detail.get_layer_params();
            }
            const dlib::tensor& get_layer_params() const override {
                return detail.get_layer_params();
            }

            bool has_bias() const override {
                return layer_has_bias(detail);
            }
            bool add_bias() override {
                return layer_add_bias(detail);
            }

            long num_filters() const override { return detail.num_filters(); }
            long nr() const override { return detail.nr(); }
            long nc() const override { return detail.nc(); }

            std::array<int,2> dilate() const override {
                return layer_dilate(detail);
            }
            std::array<int,2> stride() const override {
                return { int(detail.stride_x()), int(detail.stride_y()) };
            }
            std::array<int,2> padding() const override {
                return { int(detail.padding_x()), int(detail.padding_y()) };
            }

            void forward_const(dlib::tensor const* const* inputs,
                               std::size_t num_inputs) override {
                if (num_inputs != 1 || !inputs || !*inputs)
                    throw std::invalid_argument("single input expected");
                auto& out = allocate_output();
                detail.forward(tagged_input<0>(inputs), out);
            }

            json::array keras_array() const override {
                return layer_json(detail);
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
            void serialize_detail(std::ostream& out) const override {
                serialize(detail, out);
            }
            dlibx::parameter_format parameter_format() const override {
                return serialize_format(detail);
            }
        };
        template <long K, long NR, long NC, int... Is>
        struct layer_regular<dlib::con_<K,NR,NC,Is...> > final
            : layer_con_t<dlib::con_<K,NR,NC,Is...> > {
            using base = layer_con_t<dlib::con_<K,NR,NC,Is...> >;
            using base::base;
        };
        template <long K, long NR, long NC, int... Is>
        struct layer_regular<dlibx::lm_con_<K,NR,NC,Is...> > final
            : layer_con_t<dlibx::lm_con_<K,NR,NC,Is...> > {
            using base = layer_con_t<dlibx::lm_con_<K,NR,NC,Is...> >;
            using base::base;
        };
        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        struct layer_regular<dlibx::condw_<BM,MULT,NR,NC,Is...> > final
            : layer_con_t<dlibx::condw_<BM,MULT,NR,NC,Is...> > {
            using base = layer_con_t<dlibx::condw_<BM,MULT,NR,NC,Is...> >;
            using base::base;
        };
    }
}
