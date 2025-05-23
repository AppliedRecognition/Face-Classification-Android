#pragma once

#include "dnn_traits.hpp"

namespace dlibx {

    /** \brief Copy net from src to dest while converting certain layers.
     *
     * The dest net must have been previously setup.  Use dlibx::setup().
     *
     * Each converter object must implement the method:
     * <code>template <typename Src, typename Dest>
     *       auto operator()(const Src&, Dest&)</code>.
     * The inputs will be of type dlib::add_loss_layer, dlib::add_layer, etc.
     *
     * The return type must be a tuple of references
     * <code>std::tuple<const NextSrc&, NextDest&></code>.
     * These references are to the layers following the layers consumed.
     * One or more layers from each of the src and dest nets may be consumed.
     * One way to return these references is:
     * <code>return std::forward_as_tuple(src.subnet(), dest.subnet())</code>.
     *
     * For each layer, the converters are tried in order and the first one
     * that accepts the inputs will be used.
     * If no converter works at a particular layer, then direct assignment
     * of subnets is attempted.
     * Note that the input layer will be handled by this assignment.
     *
     * Best practice is write a converter for the specific layers that need
     * conversion and then also include the copy_layer converter (below)
     * to copy all the other trivial to copy layers.
     */
    template <typename Src, typename Dest, typename... Converters>
    void convert_from_to(const Src& src, Dest& dest, Converters&&...);


    /** \brief Direct layer copy by assignment operator.
     *
     * This converter object can handle the following layers: <ul>
     *   <li>add_loss_layer</li>
     *   <li>add_layer</li>
     *   <li>add_tag_layer</li>
     *   <li>add_skip_layer</li>
     * </ul>
     */
    struct copy_layer {
        /// copy loss layer
        template <typename Src, typename Dest>
        std::enable_if_t<
            std::is_assignable<typename Dest::loss_details_type,
                               const typename Src::loss_details_type>::value,
            std::tuple<const typename Src::subnet_type&,
                       typename Dest::subnet_type&> >
        operator()(const Src& src, Dest& dest) const {
            dest.loss_details() = src.loss_details();
            return { src.subnet(), dest.subnet() };
        }

        /// skip over tag and skip layers (any non-loss but not add_layer)
        /// no checks are made to ensure the layers are of the same type
        template <typename Src, typename Dest>
        std::enable_if_t<
            dlib::is_nonloss_layer_type<Src>::value &&
            dlib::is_nonloss_layer_type<Dest>::value &&
            !dlib::is_add_layer<Src>::value &&
            !dlib::is_add_layer<Dest>::value,
            std::tuple<const typename Src::subnet_type&,
                       typename Dest::subnet_type&> >
        operator()(const Src& src, Dest& dest) const {
            // nothing to copy or convert
            return { src.subnet(), dest.subnet() };
        }

        /// copy regular (add_layer) layer by assignment
        template <typename Src, typename Dest>
        std::enable_if_t<
            std::is_assignable<typename Dest::layer_details_type,
                               const typename Src::layer_details_type>::value,
            std::tuple<const typename Src::subnet_type&,
                       typename Dest::subnet_type&> >
        operator()(const Src& src, Dest& dest) const {
            dest.layer_details() = src.layer_details();
            return { src.subnet(), dest.subnet() };
        }
    };


    /** \brief Remove affine (or bn) layer by folding parameters into con.
     *
     * In each case of an affine or bn layer following (in the forward
     * direction) a con layer, the affine or bn parameters are folded
     * into the con layer.
     * Note that in the case of a bn layer, the effect is the same as
     * if the bn was first converted to affine.
     *
     * To use call:
     * <code>convert_from_to(src, dest, copy_layer{}, remove_affine{});</code>.
     */
    struct remove_affine {
        static auto get_params(const dlib::affine_& src) {
            // get_layer_params() is empty so have to serialize / deserialize
            std::stringstream strm;
            serialize(src, strm);
            std::string version;
            dlib::deserialize(version, strm);
            if (version != "affine_" && version != "affine_2")
                throw std::runtime_error(
                    "unknown version '" + version + "' expected affine_2");
            dlib::resizable_tensor params;
            dlib::deserialize(params, strm);
            return params;
        }

        template <typename CON>
        static void convert(CON& dest, const dlib::affine_& src) {
            filters_and_bias fb(dest);
            auto filters = fb.filters(dest);
            const auto num_filters = std::size_t(filters.num_samples());
            const auto filter_size = filters.size() / num_filters;
            assert(filters.size() == num_filters * filter_size);
            auto bias_tensor = fb.bias(dest);
            assert(bias_tensor.size() == num_filters);

            // assert(affine.get_mode() == CONV_MODE)
            const auto affine_params = get_params(src);
            assert(affine_params.size() == 2*num_filters);

            // y = x * (conv * gamma) + (bias * gamma + beta)
            auto filt = filters.host();
            auto bias = bias_tensor.host();
            auto gamma = affine_params.host();
            auto beta = gamma + num_filters;
            for (auto k = num_filters; k > 0; --k, ++bias, ++gamma, ++beta) {
                for (auto n = filter_size; n > 0; --n, ++filt) *filt *= *gamma;
                *bias = *bias * *gamma + *beta;
            }
        }

        template <typename Src, typename Dest>
        std::enable_if_t<
            dlib::is_add_layer<Dest>::value &&
            (is_con_layer<Dest>::value || is_condw_layer<Dest>::value) &&
            (is_affine_layer<Src>::value || is_bn_conv_layer<Src>::value) &&
            (is_con_layer<typename Src::subnet_type>::value ||
             is_condw_layer<typename Src::subnet_type>::value),
            std::tuple<const typename Src::subnet_type::subnet_type&,
                       typename Dest::subnet_type&> >
        operator()(const Src& src, Dest& dest) const {
            dest.layer_details() = src.subnet().layer_details();
            const dlib::affine_ affine = src.layer_details();
            convert(dest.layer_details(), affine);
            return { src.subnet().subnet(), dest.subnet() };
        }
    };

    /** \brief Convert fc layer by adding or removing bias.
     *
     * If bias needs to be added, it is initialized to zero.
     * The runtime num_outputs for src and dest must match.
     *
     * Note that conversion from dlib::fc_ to dlibx::fc_dynamic_ is
     * handled by the fc_dynamic copy constructor (in copy_layer).
     *
     * To use call:
     * <code>convert_from_to(src, dest, copy_layer{}, fc_convert{});</code>.
     */
    struct fc_convert {
        template <typename Src,
                  unsigned long K, dlib::fc_bias_mode BM, typename SUBNET,
                  typename = std::enable_if_t<is_fc_layer<Src>::value> >
        auto operator()(
            const Src& src,
            dlib::add_layer<dlib::fc_<K,BM>,SUBNET>& dest) const {

            auto& sparams = src.layer_details().get_layer_params();
            auto& dparams = dest.layer_details().get_layer_params();
            assert(sparams.nr() == 1 && sparams.nc() == 1);
            assert(dparams.nr() == 1 && dparams.nc() == 1);
            const auto k = sparams.k();
            if (k != dparams.k())
                throw std::invalid_argument("num_outputs mismatch (fc)");
            const auto n =
                std::min(dparams.num_samples(), sparams.num_samples());
            auto bias =
                std::copy_n(sparams.host(), n*k, dparams.host_write_only());
            std::fill_n(bias, (dparams.num_samples()-n)*k, 0.0f);
            return std::forward_as_tuple(src.subnet(), dest.subnet());
        }
    };
}



//// implementation

namespace dlibx {
    /// like std::tuple_element but if I is out of range, gives void type
    template <std::size_t I, typename CT, typename = void>
    struct safe_tuple_element { using type = void; };
    template <std::size_t I, typename CT>
    struct safe_tuple_element<
        I,CT,std::enable_if_t<(I < std::tuple_size<CT>::value)> >
        : std::tuple_element<I,CT> {};
    template <std::size_t I, typename CT>
    using safe_tuple_element_t = typename safe_tuple_element<I,CT>::type;


    template <typename Src, typename Dest, typename CT, std::size_t I = 0,
              typename = void>
    struct convert_net_impl;

    template <typename Src, typename Dest, typename CT, std::size_t I>
    struct convert_net_impl<
        Src, Dest, CT, I,
        std::enable_if_t<(I >= std::tuple_size<CT>::value)> > {
        inline void operator()(const Src& src, Dest& dest, CT&) const {
            dest = src;
        }
    };

    template <typename Src, typename Dest, typename CT, std::size_t I>
    struct convert_net_impl<
        Src, Dest, CT, I,
        std::enable_if_t<
            (I < std::tuple_size<CT>::value) &&
                !std::is_invocable<safe_tuple_element_t<I,CT>,
                                   const Src&,Dest&>::value> >
        : convert_net_impl<Src, Dest, CT, I+1> {};

    template <typename Src, typename Dest, typename CT, std::size_t I>
    struct convert_net_impl<
        Src, Dest, CT, I,
        std::enable_if_t<std::is_invocable<safe_tuple_element_t<I,CT>,
                                           const Src&,Dest&>::value> > {
        void operator()(const Src& src, Dest& dest, CT& ct) const {
            auto next = std::get<I>(ct)(src, dest);
            using NSrc  = std::decay_t<std::tuple_element_t<0,decltype(next)> >;
            using NDest = std::decay_t<std::tuple_element_t<1,decltype(next)> >;
            convert_net_impl<NSrc,NDest,CT>{}(
                std::get<0>(next), std::get<1>(next), ct);
        }
    };

    template <typename Src, typename Dest>
    inline void convert_from_to(const Src& src, Dest& dest) {
        dest = src;
    }

    template <typename Src, typename Dest, typename... Converters>
    void convert_from_to(const Src& src, Dest& dest, Converters&&... conv) {
        auto ct = std::forward_as_tuple(conv...);
        convert_net_impl<Src,Dest,decltype(ct)>{}(src, dest, ct);
    }
}
