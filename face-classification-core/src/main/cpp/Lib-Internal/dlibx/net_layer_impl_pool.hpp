#pragma once

#include "net_layer_impl_con.hpp" // for layer_*_construct() methods

namespace dlibx {
    namespace net {

        // **** avg_pool

        template <long NR, long NC, int SY, int SX, int PY, int PX>
        inline auto layer_code(const dlib::avg_pool_<NR,NC,SY,SX,PY,PX>&) {
            return layer_code_construct<NR,NC,SY,SX,PY,PX>("avg_pool");
        }
        template <long NR, long NC, int SY, int SX, int PY, int PX>
        constexpr auto layer_type(const dlib::avg_pool_<NR,NC,SY,SX,PY,PX>&) {
            return "avg";
        }
        template <long NR, long NC, int SY, int SX, int PY, int PX>
        inline auto layer_concise(const dlib::avg_pool_<NR,NC,SY,SX,PY,PX>&) {
            return layer_concise_construct<NR,NC,SY,SX,PY,PX>("avg");
        }
        template <long NR, long NC, int SY, int SX, int PY, int PX>
        auto layer_json(const dlib::avg_pool_<NR,NC,SY,SX,PY,PX>&) {
            json::array arr;
            json::object config;
            if (NR > 0 || NC > 0) {
                config["pool_size"] = json::array{NR,NC};
                config["strides"] = json::array{SY,SX};
                if (PY == 0 && PX == 0)
                    config["padding"] = "valid";
                else if (SY == 1 && SX == 1 && PY == NR/2 && PX == NC/2)
                    config["padding"] = "same";
                else {  // separate padding object
                    json::object zconfig;
                    zconfig["padding"] =
                        json::array{json::array{PY,PY}, json::array{PX,PX}};
                    zconfig["data_format"] = "channels_last";
                    zconfig["dtype"] = "float32";
                    zconfig["trainable"] = true;
                    json::object zobj;
                    zobj["class_name"] = "ZeroPadding2D";
                    zobj["config"] = move(zconfig);
                    zobj["name"] = "padding";
                    arr.push_back(move(zobj));
                    config["padding"] = "valid";  // no padding
                }
            }
            // else avg_pool_everything
            config["dtype"] = "float32";
            config["data_format"] = "channels_last";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = (NR > 0 || NC > 0) ?
                "AveragePooling2D" : "GlobalAveragePooling2D";
            obj["config"] = move(config);
            arr.push_back(move(obj));
            return arr;
        }


        // **** max_pool

        template <long NR, long NC, int SY, int SX, int PY, int PX>
        inline auto layer_code(const dlib::max_pool_<NR,NC,SY,SX,PY,PX>&) {
            return layer_code_construct<NR,NC,SY,SX,PY,PX>("max_pool");
        }
        template <long NR, long NC, int SY, int SX, int PY, int PX>
        constexpr auto layer_type(const dlib::max_pool_<NR,NC,SY,SX,PY,PX>&) {
            return "max";
        }
        template <long NR, long NC, int SY, int SX, int PY, int PX>
        inline auto layer_concise(const dlib::max_pool_<NR,NC,SY,SX,PY,PX>&) {
            return layer_concise_construct<NR,NC,SY,SX,PY,PX>("max");
        }
        template <long NR, long NC, int SY, int SX, int PY, int PX>
        auto layer_json(const dlib::max_pool_<NR,NC,SY,SX,PY,PX>&) {
            json::array arr;
            json::object config;
            if (NR > 0 || NC > 0) {
                config["pool_size"] = json::array{NR,NC};
                config["strides"] = json::array{SY,SX};
                if (PY == 0 && PX == 0)
                    config["padding"] = "valid";  // no padding
                else if (1 && SY == 1 && SX == 1 && PY == NR/2 && PX == NC/2)
                    config["padding"] = "same";
                else {  // separate padding object
                    json::object zconfig;
                    zconfig["padding"] =
                        json::array{json::array{PY,PY}, json::array{PX,PX}};
                    zconfig["data_format"] = "channels_last";
                    //zconfig["dtype"] = "float32";
                    zconfig["trainable"] = true;
                    json::object zobj;
                    zobj["class_name"] = "ZeroPadding2D";
                    zobj["config"] = move(zconfig);
                    zobj["name"] = "padding";
                    arr.push_back(move(zobj));
                    config["padding"] = "valid";  // no padding
                }
            }
            // else max_pool_everything
            config["dtype"] = "float32";
            config["data_format"] = "channels_last";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] =
                (NR > 0 || NC > 0) ? "MaxPooling2D" : "GlobalMaxPooling2D";
            obj["config"] = move(config);
            arr.push_back(move(obj));
            return arr;
        }

    }
}
