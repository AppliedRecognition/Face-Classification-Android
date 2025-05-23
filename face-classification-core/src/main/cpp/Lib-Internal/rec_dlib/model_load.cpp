
#include "model.ipp"
#include "model.hpp"

#include <dlibx/net_vector.hpp>
#include <dlibx/net_convert.hpp>
#include <raw_image/input_extractor.hpp>

#include <applog/core.hpp>
static constexpr const float mean_vec[] = {
    -0.1090f,0.0742f,0.0517f,-0.0375f,-0.0994f,-0.0329f,-0.0151f,-0.1079f,
    0.1378f,-0.0923f,0.2127f,-0.0365f,-0.2286f,-0.0445f,-0.0124f,0.1445f,

    -0.1405f,-0.1195f,-0.1007f,-0.0680f,0.0226f,0.0363f,0.0200f,0.0452f,
    -0.1115f,-0.3154f,-0.0861f,-0.0857f,0.0347f,-0.0633f,-0.0212f,0.0540f,

    -0.1759f,-0.0452f,0.0316f,0.0744f,-0.0404f,-0.0740f,0.1908f,0.0074f,
    -0.1750f,0.0011f,0.0608f,0.2374f,0.1846f,0.0242f,0.0188f,-0.0836f,

    0.1072f,-0.2355f,0.0457f,0.1380f,0.0863f,0.0695f,0.0580f,-0.1418f,
    0.0218f,0.1214f,-0.1886f,0.0353f,0.0607f,-0.0795f,-0.0504f,-0.0594f,

    0.2046f,0.1072f,-0.1132f,-0.1250f,0.1547f,-0.1550f,-0.0512f,0.0616f,
    -0.1190f,-0.1681f,-0.2682f,0.0425f,0.3917f,0.1305f,-0.1568f,0.0228f,

    -0.0711f,-0.0270f,0.0505f,0.0680f,-0.0632f,-0.0314f,-0.0845f,0.0344f,
    0.1964f,-0.0246f,-0.0093f,0.2210f,0.0085f,0.0091f,0.0245f,0.0508f,

    -0.0919f,-0.0210f,-0.1102f,-0.0185f,0.0413f,-0.0808f,0.0042f,0.0965f,
    -0.1852f,0.1417f,-0.0140f,-0.0215f,0.0028f,-0.0162f,-0.0834f,-0.0259f,

    0.1400f,-0.2383f,0.1883f,0.1652f,0.0180f,0.1376f,0.0564f,0.0727f,
    -0.0131f,-0.0284f,-0.1567f,-0.0831f,0.0615f,-0.0196f,0.0417f,0.0311f
};
static_assert(sizeof(mean_vec) == 128 * sizeof(float),
              "mean_vec must have size 128");

template <typename FC>
static void add_bias_to_fc(FC& fc) {
    if (fc.get_bias_mode() == dlibx::NO_BIAS) {
        fc.add_biases();
        auto bias = fc.get_biases();
        assert(bias.size() == 128);
        FILE_LOG(logINFO) << "setting recognition model bias vector";
        std::transform(std::begin(mean_vec), std::end(mean_vec), bias.host(),
                       [](float x) { return -x; });
    }
}

template <typename SUBNET>
using fc_128_no_bias = dlibx::fc_no_bias<128,SUBNET>;

template <typename NET>
static inline void add_bias_to_model(NET& model) {
    add_bias_to_fc(dlib::layer<fc_128_no_bias>(model).layer_details());
}

dlibx::net::vector rec::dlib::model_load(std::istream& in) {
    if (!in.good())
        throw std::runtime_error("failed to read recognition model");
    dlibx::net::vector model;
    if (in.peek() == 0x81)
        deserialize(model, in);
    else {
        // assume stock model
        ::dlib::loss_metric<recnet_stock::net_type> net;
        deserialize(net, in);
        add_bias_to_model(net.subnet());
        auto lv = dlibx::net::to_layers_vector(net.subnet());
        remove_affine(lv);
        model.set_layers(move(lv));
        model.meta["description"] = "stock dlib recognition model";
        if (!(model.input_extractor = raw_image::input_extractor::find("facechip150+0.25rgb")))
            throw std::runtime_error("internal input extractor failure");
    }
    return model;
}
