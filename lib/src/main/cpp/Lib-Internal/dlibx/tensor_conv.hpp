#pragma once

#include <dlib/dnn/core.h>

namespace dlibx {
    /** \brief Subclass of tt::tensor_conv.
     *
     * tt::tensor_conv is not copyable so construct as new one when copied.
     */
    struct tensor_conv : dlib::tt::tensor_conv {
        tensor_conv() = default;
        tensor_conv(const tensor_conv&)
            : dlib::tt::tensor_conv() {} // just construct a new one
        tensor_conv& operator=(const tensor_conv& other) {
            if (this != &other)
                clear();
            return *this;
        }

        /// backward() for regular (full) convolution
        void backward_conv(const dlib::tensor& filters,
                           const dlib::tensor& input,
                           dlib::tensor& output,
                           const dlib::tensor* data = nullptr,
                           dlib::tensor* filters_grad = nullptr,
                           dlib::tensor* bias_grad = nullptr);

        /// backward() for depth-wise convolution
        void backward_dw(const dlib::tensor& filters,
                         const dlib::tensor& input,
                         dlib::tensor& output,
                         const dlib::tensor* data = nullptr,
                         dlib::tensor* filters_grad = nullptr,
                         dlib::tensor* bias_grad = nullptr);
    };
}
