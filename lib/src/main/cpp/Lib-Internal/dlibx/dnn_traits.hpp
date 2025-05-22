#pragma once

#include "dnn_bias_mode.hpp"
#include <dlib/dnn/layers.h>
#include <type_traits>

namespace dlibx {

    // forward declare
    template <long,long,long,int,int,int,int,int,int> class lm_con_;
    template <bias_mode,long,long,long,int,int,int,int> class condw_;
    template <unsigned long, dlib::fc_bias_mode> class fc_dynamic_;


    /** \brief Access to input layer.
     */
    template <typename NET, typename = void>
    struct input_layer_helper {
        constexpr auto& operator()(NET& net) const { return net; }
    };
    template <typename NET>
    struct input_layer_helper<NET, std::enable_if_t<std::is_reference_v<decltype(std::declval<NET>().subnet())> > > {
        inline auto& operator()(NET& net) const {
            using SUBNET = std::remove_reference_t<decltype(net.subnet())>;
            return input_layer_helper<SUBNET>{}(net.subnet());
        }
    };
    template <typename NET>
    inline auto& input_layer(NET& net) {
        return input_layer_helper<NET>{}(net);
    }


    /** \brief Visit layer details with type selected by predicate template.
     */
    template <template<typename> typename PREDICATE, typename FUNC>
    struct visitor_for_helper {
        FUNC func;

        template <typename U>
        constexpr std::enable_if_t<!PREDICATE<std::decay_t<U> >::value>
        invoke(U&&) {}

        template <typename U>
        inline std::enable_if_t<PREDICATE<std::decay_t<U> >::value>
        invoke(U&& details) {
            func(details);
        }

        template <typename U>
        constexpr std::enable_if_t<!dlib::is_add_layer<U>::value>
        operator()(size_t, U&&) {}

        template <typename U>
        inline std::enable_if_t<dlib::is_add_layer<U>::value>
        operator()(size_t, U&& add_layer) {
            invoke(add_layer.layer_details());
        }
    };
    template <template<typename> typename PREDICATE, typename FUNC>
    inline auto visitor_for(FUNC&& func) {
        return visitor_for_helper<PREDICATE, FUNC>{std::forward<FUNC>(func)};
    }
    template <template<typename> typename PRED, typename NET, typename FUNC>
    inline auto visit_layer_details(NET&& net, FUNC&& func) {
        return dlib::visit_layers(
            std::forward<NET>(net),
            visitor_for<PRED>(std::forward<FUNC>(func)));
    }

    /** \brief Visit layer details of a specific type.
     */
    template <typename DETAILS>
    struct is_same_as {
        template <typename T>
        using type = std::is_same<T, std::decay_t<DETAILS> >;
    };
    template <typename DETAILS, typename FUNC>
    inline auto visitor_for(FUNC&& func) {
        return visitor_for_helper<is_same_as<DETAILS>::template type, FUNC>{
            std::forward<FUNC>(func)
        };
    }
    template <typename DETAILS, typename NET, typename FUNC>
    inline auto visit_layer_details(NET&& net, FUNC&& func) {
        return dlib::visit_layers(
            std::forward<NET>(net),
            visitor_for<DETAILS>(std::forward<FUNC>(func)));
    }


    /** \brief Test if layer is dlib::fc_ or dlibx::fc_dynamic_.
     */
    template <typename T> struct is_fc_layer : std::false_type {};
    template <unsigned long num_outputs, dlib::fc_bias_mode bias_mode>
    struct is_fc_layer<dlib::fc_<num_outputs,bias_mode> >
        : std::true_type {};
    template <unsigned long num_outputs, dlib::fc_bias_mode bias_mode>
    struct is_fc_layer<fc_dynamic_<num_outputs,bias_mode> >
        : std::true_type {};
    template <typename DETAILS, typename SUBNET>
    struct is_fc_layer<dlib::add_layer<DETAILS,SUBNET> >
        : is_fc_layer<DETAILS> {};


    /** \brief Test if layer is dlib::con_ or dlibx::lm_con_.
     *
     * Note that this test does not include condw.  \sa is_condw_layer
     */
    template <typename T> struct is_con_layer : std::false_type {};
    template <long num_filters, long nr, long nc,
              int sy, int sx, int py, int px>
    struct is_con_layer<dlib::con_<num_filters,nr,nc,sy,sx,py,px> >
        : std::true_type {};
    template <long num_filters, long nr, long nc,
              int sy, int sx, int py, int px, int dy, int dx>
    struct is_con_layer<lm_con_<num_filters,nr,nc,sy,sx,py,px,dy,dx> >
        : std::true_type {};
    template <typename DETAILS, typename SUBNET>
    struct is_con_layer<dlib::add_layer<DETAILS,SUBNET> >
        : is_con_layer<DETAILS> {};


    /** \brief Test if layer is dlibx::condw_.
     */
    template <typename T> struct is_condw_layer : std::false_type {};
    template <bias_mode mode, long mult, long nr, long nc,
              int sy, int sx, int py, int px>
    struct is_condw_layer<condw_<mode,mult,nr,nc,sy,sx,py,px> >
        : std::true_type {};
    template <typename DETAILS, typename SUBNET>
    struct is_condw_layer<dlib::add_layer<DETAILS,SUBNET> >
        : is_condw_layer<DETAILS> {};


    /** \brief Test if layer has void disable_bias() member method.
     *
     * Note that dlib::con_ does not have this method in older versions of dlib.
     */
    template <typename CON, typename = void>
    struct has_disable_bias : std::false_type {};
    template <typename CON>
    struct has_disable_bias<CON, std::void_t<decltype(&CON::disable_bias)> >
        : std::true_type {};
    template <typename DETAILS, typename SUBNET>
    struct has_disable_bias<dlib::add_layer<DETAILS,SUBNET> >
        : has_disable_bias<DETAILS> {};


    /** \brief Test if layer is dlib::bn_<CONV_MODE>.
     */
    template <typename T> struct is_bn_conv_layer
        : std::is_same<T,dlib::bn_<dlib::CONV_MODE> > {};
    template <typename DETAILS, typename SUBNET>
    struct is_bn_conv_layer<dlib::add_layer<DETAILS,SUBNET> >
        : is_bn_conv_layer<DETAILS> {};

    /** \brief Test if layer is dlib::affine_.
     */
    template <typename T> struct is_affine_layer
        : std::is_same<T,dlib::affine_> {};
    template <typename DETAILS, typename SUBNET>
    struct is_affine_layer<dlib::add_layer<DETAILS,SUBNET> >
        : is_affine_layer<DETAILS> {};

    /** \brief Test if net has subnet() method returning a reference.
     */
    template <typename, typename = void>
    struct has_subnet : std::false_type {};
    template <typename NET>
    struct has_subnet<NET, std::enable_if_t<std::is_reference_v<decltype(std::declval<NET&>().subnet())> > > : std::true_type {};

    /** \brief Force setup of a net by running an image through.
     *
     * The args are passed to the constructor for the input_type.
     * For dlib::matrix these are rows,cols (ie. height,width).
     *
     * The image pixels are not initialized but this shouldn't matter
     * as setup() depends on the dimensions only.
     *
     * Also, net.clean() is called after setup.
     */
    template <typename NET, typename... Args>
    void setup(NET& net, Args&&... args) {
        using input_type = typename NET::input_type;
        input_type img(std::forward<Args>(args)...);
        net(img);
        net.clean();
    }


    /** \brief Method to retrieve alias tensors for the filters and bias
     * of a convolution.
     */
    struct filters_and_bias {
        dlib::alias_tensor filters_alias, bias_alias;

        template <typename CON>
        inline auto filters(CON&& con) const {
            return filters_alias(con.get_layer_params(),0);
        }
        template <typename CON>
        inline auto bias(CON&& con) const {
            return bias_alias(con.get_layer_params(),filters_alias.size());
        }

        template <typename CON>
        filters_and_bias(
            const CON& con,
            std::enable_if_t<is_con_layer<CON>::value>* = nullptr) {
            const auto params_size = long(con.get_layer_params().size());
            const auto channel_size = con.nr() * con.nc();
            const auto num_filters = con.num_filters();
            const auto input_channels =
                (params_size / num_filters - 1) / channel_size;
            if (params_size <= 0 ||
                params_size != num_filters * (input_channels*channel_size+1))
                throw std::invalid_argument(
                    "convolution has invalid number of parameters");
            filters_alias =
                dlib::alias_tensor(num_filters,input_channels,con.nr(),con.nc());
            bias_alias = dlib::alias_tensor(1,num_filters);
        }

        template <typename CON>
        filters_and_bias(
            const CON& con,
            std::enable_if_t<is_condw_layer<CON>::value>* = nullptr) {
            const auto params_size = long(con.get_layer_params().size());
            const auto channel_size = con.nr() * con.nc();
            const auto has_bias = con.get_bias_mode() == HAS_BIAS ? 1 : 0;
            const auto num_filters = params_size / (channel_size + has_bias);
            if (params_size <= 0 ||
                params_size != num_filters * (channel_size + has_bias))
                throw std::invalid_argument(
                    "depth-wise convolution has invalid number of parameters");
            filters_alias =
                dlib::alias_tensor(num_filters,1,con.nr(),con.nc());
            bias_alias = dlib::alias_tensor(has_bias,num_filters);
        }
    };

    template <typename CON>
    auto get_filters_and_bias(CON&& con) {
        filters_and_bias fb(con);
        return std::make_pair(fb.filters(con), fb.bias(con));
    }
}


