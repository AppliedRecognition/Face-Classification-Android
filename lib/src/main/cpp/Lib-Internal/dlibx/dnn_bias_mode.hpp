#pragma once

#include <dlib/dnn/layers.h>

namespace dlibx {
    using bias_mode = dlib::fc_bias_mode;
    static constexpr auto NO_BIAS = dlib::FC_NO_BIAS;
    static constexpr auto HAS_BIAS = dlib::FC_HAS_BIAS;
}
