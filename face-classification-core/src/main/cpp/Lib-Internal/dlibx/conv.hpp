#pragma once

#include "tensor.hpp"
#include <memory>

namespace dlibx {

    /** \brief Copy input tensor to new tensor with zero padding added.
     *
     * The version without a specified output uses an internal thread_local
     * object, so care must be taken to ensure it's not used a second time
     * before use of the first output is complete.
     */
    const dlib::tensor&
    apply_padding(const dlib::tensor& input, dlib::resizable_tensor& output,
                  int top, int left, int bottom, int right);
    inline const dlib::tensor&
    apply_padding(const dlib::tensor& input, dlib::resizable_tensor& output,
                  int top_bottom, int left_right) {
        return apply_padding(input, output,
                             top_bottom, left_right,
                             top_bottom, left_right);
    }
    std::shared_ptr<const dlib::tensor>
    apply_padding(const dlib::tensor& input,
                  int top, int left, int bottom, int right);
    inline std::shared_ptr<const dlib::tensor>
    apply_padding(const dlib::tensor& input, int top_bottom, int left_right) {
        return apply_padding(input,
                             top_bottom, left_right,
                             top_bottom, left_right);
    }

    
    /** \brief Pointwise and full/general convolution inference.
     *
     * Floating point forward direction only.
     */
    class forward_conv {
        struct internal;
        std::unique_ptr<internal> state;
        void(internal::*m)(const dlib::tensor& src, dlib::resizable_tensor&);

        forward_conv(forward_conv&&) = delete;
        forward_conv& operator=(forward_conv&&) = delete;

    public:
        forward_conv();
        ~forward_conv();
        void reset();

        // copy and assign produce an empty (not setup) object
        forward_conv(const forward_conv&);
        forward_conv& operator=(const forward_conv&);

        void setup(int nr, int nc, int dy, int dx,
                   int sy, int sx, int py, int px,
                   const dlib::tensor& filters);

        inline explicit operator bool() const { return state.get(); }

        inline void operator()(
            const dlib::tensor& src, dlib::resizable_tensor& dest) const {
            ((*state).*m)(src,dest);
        }
    };


    /** \brief Depthwise convolution inference.
     *
     * Floating point forward direction only.
     */
    class forward_convdw {
        struct internal;
        std::unique_ptr<internal> state;
        void(internal::*m)(const dlib::tensor& src, dlib::resizable_tensor&);

        forward_convdw(forward_convdw&&) = delete;
        forward_convdw& operator=(forward_convdw&&) = delete;

    public:
        forward_convdw();
        ~forward_convdw();
        void reset();

        // copy and assign produce an empty (not setup) object
        forward_convdw(const forward_convdw&);
        forward_convdw& operator=(const forward_convdw&);

        void setup(int nr, int nc, int dy, int dx,
                   int sy, int sx, int py, int px,
                   const dlib::tensor& filters);

        inline explicit operator bool() const { return state.get(); }

        inline void operator()(
            const dlib::tensor& src, dlib::resizable_tensor& dest) const {
            ((*state).*m)(src,dest);
        }
    };


}
