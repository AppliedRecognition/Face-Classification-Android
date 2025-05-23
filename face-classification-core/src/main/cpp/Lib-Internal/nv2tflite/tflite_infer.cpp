
#include "tflite_infer.hpp"

#include <raw_image/pixels.hpp>

#include <tensorflow/lite/kernels/register.h>

std::vector<dlib::resizable_tensor>
tflite::infer(const tflite::FlatBufferModel& model,
              const raw_image::plane& img) {

    // build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(model, resolver)(&interpreter);

    // resize input tensors
    interpreter->AllocateTensors();
    
    // copy input
    auto& inputs = interpreter->inputs();
    assert(inputs.size() == 1);
    const auto input = inputs[0];
    auto const* dims = interpreter->tensor(input)->dims;
    assert(dims && dims->size == 4);
    assert(dims->data[0] == 1 &&
           dims->data[1] == long(img.height) &&
           dims->data[2] == long(img.width) &&
           dims->data[3] == 3); // channels
    {
        float* dest = interpreter->typed_input_tensor<float>(0);
        for (auto&& line : raw_image::pixels_bpp<3>(img))
            for (auto& px : line) {
                *dest++ = float(px[0]-128)/128;
                *dest++ = float(px[1]-128)/128;
                *dest++ = float(px[2]-128)/128;
            }
    }

    interpreter->Invoke();

    auto& outputs = interpreter->outputs();
    std::vector<dlib::resizable_tensor> out;
    out.reserve(outputs.size());
    for (unsigned i = 0; i < outputs.size(); ++i) {
        const auto j = outputs[i];
        auto& srct = *interpreter->tensor(j);
        auto const* dims = srct.dims;
        assert(dims && 2 <= dims->size && dims->size <= 4);
        auto& destt = out.emplace_back(
            dims->data[0], dims->data[1],
            3 <= dims->size ? dims->data[2] : 1,
            4 <= dims->size ? dims->data[3] : 1);
        float const* src = interpreter->typed_output_tensor<float>(int(i));
        float* dest = destt.host_write_only();
        std::copy_n(src, destt.size(), dest);
    }
    return out;
}
