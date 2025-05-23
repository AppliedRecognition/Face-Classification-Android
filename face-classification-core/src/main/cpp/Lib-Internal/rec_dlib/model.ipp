
#include <dlibx/dnn_lmcon.hpp>
#include <dlibx/dnn_fc_dynamic.hpp>
#include <dlibx/dnn_input_generic_image.hpp>
#include <dlibx/resnet.hpp>

namespace {
    /// for training (no bias)
    struct traits_train {
        template <long K, long NR, long NC, int SY, int SX, typename SUBNET>
        using con = dlibx::lmcon<K,NR,NC,SY,SX,SUBNET>;

        template <typename SUBNET>
        using bn = dlib::bn_con<SUBNET>;

        template <unsigned long K, typename SUBNET>
        using fc = dlib::fc_no_bias<K,SUBNET>;

        using input = dlib::input_rgb_image_sized<150>;
    };

    /// residual net for face recognition (base class)
    struct recnet_base {
        static constexpr auto structure =
            "128=256<r256|r256|r256<r128|r128|r128"
            "<r64|r64|r64|r64<r32|r32|r32|r32<m32<c7x7x3";

        using pixel_type = dlib::rgb_pixel;
        using image_type = dlib::matrix<pixel_type>;
        using input_layer = dlibx::input_generic_image<image_type>;

        using train_type =
            dlib::loss_metric<dlibx::resnet::net<traits_train> >;

        template <typename NET>
        static inline auto& stored_net(NET& net) { return net; }
    };

    /// for custom models (no affine and with bias)
    struct traits_custom {
        template <long K, long NR, long NC, int SY, int SX, typename SUBNET>
        using con = dlibx::lmcon<K,NR,NC,SY,SX,SUBNET>;

        template <typename SUBNET>
        using bn = void;

        template <unsigned long K, typename SUBNET>
        using fc = dlib::fc<K,SUBNET>;

        using input = dlib::input_rgb_image_sized<150>;
    };
    struct recnet_custom : recnet_base {
        static constexpr auto code = "bypwjxy";
        using net_type = dlibx::resnet::net<traits_custom>;
    };

    /// stock dlib neural net (using affine and dynamic bias)
    /// bias is not present when loaded and added after load
    struct traits_stock {
        template <long K, long NR, long NC, int SY, int SX, typename SUBNET>
        using con = dlibx::lmcon<K,NR,NC,SY,SX,SUBNET>;

        template <typename SUBNET>
        using bn = dlib::affine<SUBNET>;

        template <unsigned long K, typename SUBNET>
        using fc = dlibx::fc_no_bias<K,SUBNET>;

        using input = dlib::input_rgb_image_sized<150>;
    };
    struct recnet_stock : recnet_base {
        static constexpr auto code = "b76w43o";
        using net_type = dlibx::resnet::net<traits_stock>;
    };
}
