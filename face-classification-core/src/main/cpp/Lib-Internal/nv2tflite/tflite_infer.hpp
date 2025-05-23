#pragma once

#include <dlibx/tensor.hpp>
#include <raw_image/types.hpp>
#include <tensorflow/lite/model_builder.h>

namespace tflite {
    std::vector<dlib::resizable_tensor>
    infer(const tflite::FlatBufferModel& model,
          const raw_image::plane& img);
}
