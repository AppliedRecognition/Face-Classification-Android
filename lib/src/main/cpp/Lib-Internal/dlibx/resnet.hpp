#pragma once

/* Recognition model with options:
 *  + selection of con layer
 *  + with bn (batch normalize), affine or neither
 *  + final layer is fc or fc_no_bias, and either dlib or dlibx versions
 */

#include <dlib/dnn.h>

namespace dlibx {
namespace resnet {
    using namespace dlib;

    template <typename TRAITS, int MAX_CHANNELS = 256>
    struct layers {
        template <long K, long NR, long NC, int SY, int SX, typename SUBNET>
        using bncon = std::conditional_t<
            std::is_void<typename TRAITS::template bn<SUBNET> >::value,
            typename TRAITS::template con<K,NR,NC,SY,SX,SUBNET>,
            typename TRAITS::template bn<typename TRAITS::template con<K,NR,NC,SY,SX,SUBNET> > >;

        template <int K, int stride, typename SUBNET>
        using block = bncon<K,3,3,1,1,relu<bncon<K,3,3,stride,stride,SUBNET> > >;

        template <int N, typename SUBNET>
        using res = relu<add_prev1<block<N,1,tag1<SUBNET> > > >;

        template <int N, typename SUBNET>
        using res_down = relu<add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,2,tag1<SUBNET> > > > > > >;

        static constexpr auto K0 = std::min(256,MAX_CHANNELS);
        template <typename SUBNET>
        using level0 = res_down<K0,SUBNET>;

        static constexpr auto K1 = std::min(256,MAX_CHANNELS);
        template <typename SUBNET>
        using level1 = res<K1,res<K1,res_down<K1,SUBNET> > >;

        static constexpr auto K2 = std::min(128,MAX_CHANNELS);
        template <typename SUBNET>
        using level2 = res<K2,res<K2,res_down<K2,SUBNET> > >;

        static constexpr auto K3 = std::min(64,MAX_CHANNELS);
        template <typename SUBNET>
        using level3 = res<K3,res<K3,res<K3,res_down<K3,SUBNET> > > >;

        static constexpr auto K4 = std::min(32,MAX_CHANNELS);
        template <typename SUBNET>
        using level4 = res<K4,res<K4,res<K4,SUBNET> > >;

        using input = max_pool<3,3,2,2,relu<bncon<K4,7,7,2,2,typename TRAITS::input> > >;
    };

    template <typename LAYERS>
    using core = avg_pool_everything<
        typename LAYERS::template level0<
            typename LAYERS::template level1<
                typename LAYERS::template level2<
                    typename LAYERS::template level3<
                        typename LAYERS::template level4<
                            typename LAYERS::input>
                        >
                    >
                >
            >
        >;

    template <typename TRAITS>
    using net = typename TRAITS::template fc<128,core<layers<TRAITS> > >;

    /* Example traits class:
     *
     * struct traits {
     *     template <long K, long NR, long NC, int SY, int SX, typename SUBNET>
     *     using con = dlib::con<K,NR,NC,SY,SX,SUBNET>;
     *
     *     template <typename SUBNET>
     *     using bn = dlib::affine<SUBNET>;
     *
     *     template <unsigned long K, typename SUBNET>
     *     using fc = dlib::fc_no_bias<K,SUBNET>;
     *
     *     using input = dlib::input_rgb_image_sized<150>;
     * };
     */
}
}
