#pragma once

#include "net_layer_impl.hpp"
#include "dnn_input_frames.hpp"

namespace dlibx {
    namespace net {

        /** \brief State of conversion process.
         */
        struct conversion_state {
            std::vector<std::unique_ptr<layer> > layers;

            std::map<std::string_view, unsigned> name_map;

            inline auto from0(std::string_view key) {
                if (const auto i = name_map[key]++)
                    return std::string(key) + '_' + std::to_string(i);
                else
                    return std::string(key);
            }

            inline auto from1(std::string_view key) {
                return std::string(key) + '_' + std::to_string(++name_map[key]);
            }
            
            std::map<unsigned long, std::string> tag_map;
            std::string input;
        };
        

        // input layers
        
        template <typename T>
        struct is_input_layer : std::false_type {};

        template <typename T>
        struct is_input_layer<dlib::input<T> > : std::true_type {};
        template <>
        struct is_input_layer<dlib::input_rgb_image> : std::true_type {};
        template <std::size_t NR, std::size_t NC>
        struct is_input_layer<dlib::input_rgb_image_sized<NR,NC> >
            : std::true_type {};
        template <typename T>
        struct is_input_layer<dlibx::input_generic_image<T> >
            : std::true_type {};
        template <std::size_t N>
        struct is_input_layer<dlibx::input_frames<N> >
            : std::true_type {};
    
        template <typename INPUT>
        std::enable_if_t<is_input_layer<INPUT>::value>
        populate_layers(const INPUT& input, conversion_state& s) {
            s.layers.emplace_back(std::make_unique<layer_input<INPUT> >(input));
            s.input = s.layers.back()->name = s.from0("input_image");
        }


        // tag and skip layers (declaration)

        template <unsigned long ID, typename SUBNET>
        void populate_layers(const dlib::add_tag_layer<ID,SUBNET>& net,
                             conversion_state& s);
        template <template<typename> class TAG, typename SUBNET>
        void populate_layers(const dlib::add_skip_layer<TAG,SUBNET>& net,
                             conversion_state& s);


        // regular layers

        template <typename DETAILS>
        struct populate_name {
            static void set(conversion_state& s) {
                s.layers.back()->name = s.from1("layer");
            }
        };
        template <long K, long NR, long NC, int SY, int SX, int PY, int PX>
        struct populate_name<dlib::con_<K,NR,NC,SY,SX,PY,PX> > {
            static void set(conversion_state& s) {
                s.layers.back()->name = s.from1("conv");
            }
        };
        template <long K, long NR, long NC, int SY, int SX, int PY, int PX>
        struct populate_name<dlibx::lm_con_<K,NR,NC,SY,SX,PY,PX> > {
            static void set(conversion_state& s) {
                s.layers.back()->name = s.from1("conv");
            }
        };
        template <>
        struct populate_name<dlib::affine_> {
            static void set(conversion_state& s) {
                s.layers.back()->name = s.from1("sc");
            }
        };
        template <>
        struct populate_name<dlib::relu_> {
            static void set(conversion_state& s) {
                s.layers.back()->name = s.from0("activation");
            }
        };
        template <typename... FUNCS>
        struct populate_name<dlibx::lambda_<FUNCS...> > {
            static void set(conversion_state& s) {
                s.layers.back()->name = s.from0("lambda");
            }
        };
        template <long NR, long NC, int SY, int SX, int PY, int PX>
        struct populate_name<dlib::max_pool_<NR,NC,SY,SX,PY,PX> > {
            static void set(conversion_state& s) {
                s.layers.back()->name = s.from0("max_pooling2d");
            }
        };
        template <long NR, long NC, int SY, int SX, int PY, int PX>
        struct populate_name<dlib::avg_pool_<NR,NC,SY,SX,PY,PX> > {
            static void set(conversion_state& s) {
                s.layers.back()->name = s.from0("average_pooling2d");
            }
        };

        template <template<typename> class... TAGS>
        struct populate_inbound {
            static constexpr void set(conversion_state&) {} // base case
        };
        template <template<typename> class TAG,
                  template<typename> class... TAGS>
        struct populate_inbound<TAG,TAGS...> {
            static constexpr auto ID = dlib::tag_id<TAG>::id;
            static inline void set(conversion_state& s) {
                const auto& tag = s.tag_map[ID];
                if (tag.empty())
                    throw std::runtime_error("tag not found");
                s.layers.back()->inbound.emplace_back(tag);
                populate_inbound<TAGS...>::set(s);
            }
        };

        template <template<typename> class TAG>
        struct populate_name<dlib::add_prev_<TAG> > {
            static void set(conversion_state& s) {
                auto& layer = *s.layers.back();
                layer.inbound.reserve(2);
                layer.inbound.emplace_back(s.input);
                populate_inbound<TAG>::set(s);
                layer.name = s.from0("add");
            }
        };
        template <template<typename> class TAG>
        struct populate_name<dlib::mult_prev_<TAG> > {
            static void set(conversion_state& s) {
                auto& layer = *s.layers.back();
                layer.inbound.reserve(2);
                layer.inbound.emplace_back(s.input);
                populate_inbound<TAG>::set(s);
                layer.name = s.from0("mult");
            }
        };
        template <template<typename> class... TAGS>
        struct populate_name<dlib::concat_<TAGS...> > {
            static void set(conversion_state& s) {
                auto& layer = *s.layers.back();
                layer.inbound.reserve(sizeof...(TAGS));
                populate_inbound<TAGS...>::set(s);
                layer.name = s.from0("concat");
            }
        };

        template <typename DETAILS, typename SUBNET>
        void populate_layers(const dlib::add_layer<DETAILS,SUBNET>& net,
                             conversion_state& s) {
            populate_layers(net.subnet(), s);
            s.layers.emplace_back(
                std::make_unique<layer_generic<DETAILS> >(net.layer_details()));
            auto& layer = *s.layers.back();
            populate_name<DETAILS>::set(s);
            if (layer.inbound.empty())
                layer.inbound.push_back(s.input);
            s.input = layer.name;
        }


        // tag and skip layers (implementation)

        template <unsigned long ID, typename SUBNET>
        void populate_layers(const dlib::add_tag_layer<ID,SUBNET>& net,
                             conversion_state& s) {
            populate_layers(net.subnet(), s);
            s.tag_map[ID] = s.input;
        }

        template <template<typename> class TAG, typename SUBNET>
        void populate_layers(const dlib::add_skip_layer<TAG,SUBNET>& net,
                             conversion_state& s) {
            populate_layers(net.subnet(), s);
            static constexpr auto id = dlib::tag_id<TAG>::id;
            s.input = s.tag_map[id];
            if (s.input.empty())
                throw std::runtime_error("tag not found");
        }
        
        
        /** \brief Construct vector of net::layer objects from
         * native dlib dnn class.
         */
        template <typename SUBNET>
        auto to_layers_vector(const SUBNET& net) {
            conversion_state s;
            s.layers.reserve(SUBNET::num_layers);
            populate_layers(net, s);
            return move(s.layers);
        }


        /** \brief Convert batch normalize layers to affine.
         *
         * Iterator must dereference to layer_ptr&.
         */
        template <typename ITER>
        auto convert_to_affine(ITER first, ITER last) {
            std::size_t converted = 0;
            for ( ; first != last; ++first) {
                layer_ptr& layer = *first;
                if (auto bn = dynamic_cast<layer_bncon*>(layer.get())) {
                    auto nl = std::make_unique<layer_affine>(bn->detail);
                    nl->name = layer->name;
                    nl->inbound = layer->inbound;
                    layer = move(nl);
                    ++converted;
                }
                else if (auto bn = dynamic_cast<layer_bnfc*>(layer.get())) {
                    auto nl = std::make_unique<layer_affine>(bn->detail);
                    nl->name = layer->name;
                    nl->inbound = layer->inbound;
                    layer = move(nl);
                    ++converted;
                }
            }
            return converted;
        }


        /** \brief Remove affine layers following con layer.
         *
         * The affine parameters are "folded" into the con layer.
         * Layers that depend on the affine's output (inbound) are updated
         * to take input from the con layer instead.
         *
         * This method will also remove multiply layers the same way as
         * they are just special cases of affine.
         */
        void remove_affine(std::vector<layer_ptr>& layers);

        /** \brief Remove all invdropout layers.
         */
        void remove_dropout(std::vector<layer_ptr>& layers);


        /** \brief Serialize to dlib native format.
         *
         * Use this method to convert from net::vector to
         * dlib native class structure.
         * First serialize and then deserialize to native class.
         *
         * This method requires the mapping of the layers to be regular in
         * the sense that for every middle node, the first inbound_node must
         * be the immediately preceeding node, and the first outbound_node
         * must be the immediately following node.
         *
         * \return total number of layers including tags and input
         */
        unsigned serialize_native(
            stdx::span<const layer_ptr> layers, std::ostream& out);
    }
}
