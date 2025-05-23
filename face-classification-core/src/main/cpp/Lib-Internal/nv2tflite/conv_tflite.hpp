#pragma once

#include "conv_tools.hpp"

#include <dlibx/net_layer.hpp>
#include <tensorflow/lite/model_builder.h>

#include <applog/levels.hpp>

#include <filesystem>

namespace conv {

    inline auto builtin_code(const tflite::OperatorCode& oc) {
        if (oc.builtin_code() != 0)
            return oc.builtin_code();
        return tflite::BuiltinOperator(oc.deprecated_builtin_code());
    }

    void dequantize(const flatbuffers::Vector<uint8_t>& src_data,
                    tflite::TensorType src_type, unsigned bytes_per_el,
                    const shape_type& shape, dlib::resizable_tensor& dest);

    struct layer_args {
        const shape_type& out_shape;
        const std::vector<shape_type>& in_shapes;
        const std::vector<dlib::tensor const*>& params;
        const tflite::Operator& op;
    };

    dlibx::net::layer_ptr
    make_layer(const tflite::OperatorCode& opcode,
               const layer_args& args);


    /** \brief Tflite model unpacked.
     */
    class tflite_model {
        std::unique_ptr<tflite::impl::FlatBufferModel> fbmodel_ptr;

    public:
        template <typename T>
        using OffsetVector = flatbuffers::Vector<flatbuffers::Offset<T> >;

        tflite::impl::FlatBufferModel const& fbmodel;
        tflite::Model const& model;

        OffsetVector<tflite::Buffer> const& buffers;
        OffsetVector<tflite::OperatorCode> const& opcodes;

        // note: tflite::SubGraph is different from tflite::Subgraph
        tflite::SubGraph const& subgraph;
        OffsetVector<tflite::Tensor> const& sg_tensors;
        OffsetVector<tflite::Operator> const& sg_operators;

        unsigned input_tensor_index = 0;
        shape_type input_shape;

        std::vector<unsigned> output_tensor_index;

        // decoded model parameters
        std::vector<dlib::resizable_tensor> tensors;

        tflite_model(const std::filesystem::path& path);

        void log_metadata(applog::log_level = logINFO) const;

        unsigned copy_float32_and_int32_params();
        unsigned dequantize_params();
    };
}
