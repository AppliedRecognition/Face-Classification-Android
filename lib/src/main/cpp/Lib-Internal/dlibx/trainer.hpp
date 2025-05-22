#pragma once

#include <dlib/dnn/trainer.h>
#include "solvers.hpp"
#include "dnn_traits.hpp"

namespace dlibx {

    /** \brief Same as dlib::dnn_trainer but with dlibx::sgd as default.
     */
    template <typename net_type, typename solver_type = sgd>
    using dnn_trainer = dlib::dnn_trainer<net_type,solver_type>;


    /** \brief Helper for disable_bias_learning() below.
     */
    template <typename NET, typename SUBNET, typename = void>
    struct disable_bias_learning_helper {
        constexpr auto operator()(SUBNET&) const { return 0; }
    };
    template <typename NET, typename SUBNET>
    struct disable_bias_learning_helper<
        NET,SUBNET,std::enable_if_t<is_bn_conv_layer<NET>::value &&
                                    is_con_layer<SUBNET>::value> > {
        inline auto operator()(SUBNET& s) const {
            s.layer_details().set_bias_learning_rate_multiplier(0);
            return 1;
        }
    };

    /** \brief Set learning rate to zero for the biases of any convolution
     * which is immediately followed by batch normalization
     * (in the forward direction).
     *
     * \returns count of consolution layers altered
     */
    template <typename NET>
    constexpr std::enable_if_t<!has_subnet<NET>::value, unsigned>
    disable_bias_learning(NET&) {
        return 0;
    }
    template <typename NET>
    inline std::enable_if_t<has_subnet<NET>::value, unsigned>
    disable_bias_learning(NET& net) {
        using SUBNET = std::decay_t<decltype(net.subnet())>;
        disable_bias_learning_helper<NET,SUBNET> h{};
        return h(net.subnet()) + disable_bias_learning(net.subnet());
    }


    /** \brief Helper for disable_bias() below.
     */
    template <typename NET, typename SUBNET, typename = void>
    struct disable_bias_helper {
        constexpr auto operator()(SUBNET&) const { return 0; }
    };
    template <typename NET, typename SUBNET>
    struct disable_bias_helper<
        NET,SUBNET,std::enable_if_t<is_bn_conv_layer<NET>::value &&
                                    has_disable_bias<SUBNET>::value> > {
        inline auto operator()(SUBNET& s) const {
            s.layer_details().disable_bias();
            return 1;
        }
    };

    /** \brief Invoke disable_bias() member methods on any convolution
     * which is immediately followed by batch normalization
     * (in the forward direction).
     *
     * \returns count of consolution layers altered
     */
    template <typename NET>
    constexpr std::enable_if_t<!has_subnet<NET>::value, unsigned>
    disable_bias(NET&) {
        return 0;
    }
    template <typename NET>
    inline std::enable_if_t<has_subnet<NET>::value, unsigned>
    disable_bias(NET& net) {
        using SUBNET = std::decay_t<decltype(net.subnet())>;
        disable_bias_helper<NET,SUBNET> h{};
        return h(net.subnet()) + dlibx::disable_bias(net.subnet());
    }
}
