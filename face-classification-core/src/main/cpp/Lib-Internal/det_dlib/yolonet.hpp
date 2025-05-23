#pragma once

#include <dlibx/dnn_condw.hpp>
#include <dlibx/dnn_lmcon.hpp>
#include <dlib/dnn.h>

namespace det {
    namespace yolo {

        /** \brief Tiny face detector from face-api.js.
         *
         * Model file must include prelu parameter 0.1f and
         * mean rgb values [117.0, 114.7, 97.4].
         * Neither of these are the dlib default values!
         */
        template <typename TRAITS>
        struct tiny_face_detector_layers {
            template <typename SUBNET>
            using leaky = dlib::prelu<SUBNET>;

            template <long K, typename SUBNET>
            using con1 = typename TRAITS::template con<K,1,1,1,1,SUBNET>;

            template <long K, typename SUBNET>
            using con3 = typename TRAITS::template con<K,3,3,1,1,SUBNET>;

            template <typename SUBNET>
            using pool2 = dlib::max_pool<2,2,2,2,SUBNET>;

            template <long K, typename SUBNET>
            using dscon3 =
                leaky<con1<K,dlibx::condw_no_bias<3,3,1,1,pool2<SUBNET> > > >;

            template <typename SUBNET>
            using reduce =
                dscon3<512,dscon3<256,dscon3<128,dscon3<64,dscon3<32,SUBNET> > > > >;

            template <typename SUBNET>
            using pool1 = dlib::add_layer<dlib::max_pool_<2,2,1,1>,SUBNET>;

            using input = typename TRAITS::input;
        
            using core = con1<25,pool1<reduce<leaky<con3<16,input> > > > >;
        };

        template <typename TRAITS>
        using tiny_face_detector =
            typename tiny_face_detector_layers<TRAITS>::core;

        /// traits classes:

        struct dlib_con {
            template <long K, long NR, long NC, int SY, int SX, typename SUBNET>
            using con = dlib::con<K,NR,NC,SY,SX,SUBNET>;

            using input = dlib::input_rgb_image;
        };

        struct lmcon {
            template <long K, long NR, long NC, int SY, int SX, typename SUBNET>
            using con = dlibx::lmcon<K,NR,NC,SY,SX,SUBNET>;

            using input = dlib::input_rgb_image;
        };

        // for face-api.js tiny face detector
        constexpr float tiny_face_detector_boxes[][2] = {
            { 1.603231f, 2.094468f },
            { 6.041143f, 7.080126f },
            { 2.882459f, 3.518061f },
            { 4.266906f, 5.178857f },
            { 9.041765f, 10.66308f }
        };
    }
}
