#pragma once

#include "tensor.hpp"
#include <raw_image/types.hpp>
#include <json/types.hpp>
#include <stdext/arg.hpp>
#include <stdext/span.hpp>
#include <stdext/forward_iterator.hpp>
#include <string_view>


namespace dlibx {
    /// forward declare -- see bfloat16.hpp for definition
    enum class parameter_format;

    namespace net {

        // forward declare
        class layer;
        using layer_ptr = std::unique_ptr<layer>;


        /** \brief Abstract base class for neural net layer implementation.
         */
        class layer {
        public:
            virtual ~layer() = default;


            /** \brief Unique name for layer.
             */
            std::string name;


            /** \brief Name of layers from which to draw inputs.
             *
             * This vector must be empty for the front (input) layer and
             * non-empty for all other layers.
             * The inbound layers must all refer to layers that came earlier
             * (towards the front) in the vector of layers.
             */
            std::vector<std::string> inbound;


            /** \brief Pointers to inbound layers.
             *
             * This vector is not serialized nor restored on deserialize.
             * It is filled in by map_layers().
             */
            std::vector<layer*> inbound_nodes;


            /** \brief Pointers to layers that will receive the output from 
             * this layer.
             *
             * This vector is filled in my map_layers().
             */
            std::vector<layer*> outbound_nodes;


            /** \brief Construct copy.
             */
            layer_ptr copy() const;


            /** \brief Layer code.
             *
             * The layer code uniquely identifies the subclass required
             * to implement it.
             */
            virtual std::string code() const = 0;


            /** \brief Layer description object.
             *
             * type is similar to code() but doesn't include compile time
             * constants.  For example, where code() may be "con_3_3_1_1",
             * type would be "con".
             *
             * concise is a complete desciption of what the layer does.
             * For example, "bias|con3*2|pad1" or "dw1x3|pad0+1".
             *
             * Only convolution and fc layers have a non-zero output_channels.
             */
            struct description {
                std::string_view type;
                std::string concise;
                unsigned long output_channels;
                unsigned long parameters;
            };
            /** \brief Get layer description.
             */
            virtual description layer_description() const = 0;


            /** \brief Get type of layer and size of output tensor (if known).
             *
             * The type is similar to code() but doesn't not have to include
             * compile time constants.  For example, where code() may return
             * "con_3_3_1_1", the type will simply be "con".
             *
             * The size of the output is generally the number of channels
             * output, not the size of the tensor.  
             * Only convolution and fc layers have a non-zero output size.
             *
             * \sa last_output() to obtain the dimensions of the last
             *     sample run through
             */
            inline auto layer_type_and_output_size() const {
                const auto d = layer_description();
                return std::pair(d.type, d.output_channels);
            }


            /** \brief Generate a concise description of neural net structure.
             *
             * \pre map_layers() has been called on the net
             *
             * Provides a description starting at this node as the output
             * node and following inbound_nodes towards the input.
             * Call this method on the actual output node of the neural net
             * to obtain a concise description of the entire net.
             *
             * Output tensor dimensions, determined by calling last_output(),
             * will be included if a sample has been run through the net.
             * Otherwise, layer_type_and_output_size() will be used to
             * determine the number of output channels for convolution and
             * fully connected layers.
             */
            std::string concise() const;


            /** \brief Access to layer parameters.
             *
             * The const version will return an empty tensor in the case
             * where the layer does not have params.
             * The non-const version will throw an exception in this case.
             */
            virtual dlib::tensor& get_layer_params();
            virtual const dlib::tensor& get_layer_params() const;


            /** \brief For input layer only, convert multiple images to tensor.
             *
             * Note that each span<plane> is a single possibily multi-frame
             * input sample, not a multi-plane image such as Y8 + VU16.  
             * Each input frame must be a single plane.
             */
            dlib::tensor& forward(
                stdx::forward_iterator<stdx::span<const raw_image::plane> > first,
                stdx::forward_iterator<stdx::span<const raw_image::plane> > last);

            /** \brief For computational layers.
             *
             * Inputs are drawn from inbound_nodes as filled in by
             * map_layers().
             */
            dlib::tensor& forward();

            /** \brief For computational layers.
             *
             * This method is for testing individual layers that have only
             * a single input.
             * Class net::vector does not use this method.
             */
            dlib::tensor& forward(const dlib::tensor& input);


            /** \brief Allocate output tensor as copy of input.
             *
             * This method may be used to override the normal input image
             * to tensor conversion or to skip computation of any other layer.
             */
            dlib::tensor& assign_output(const dlib::tensor& input);


            /** \brief Access to output of last forward() call.
             *
             * If the next layer in the stack can operate in place on
             * the output of this layer, then this method will return
             * the output of that layer (not the output of this layer).
             * However, note that layers that operate in place do not change
             * the dimensions of the tensor, so this method can be used to
             * determine them.
             */
            const dlib::tensor& last_output() const;


            /** \brief Deserialize.
             *
             * This method will first deserialize name, inbound and
             * the code identifying the type (class) of layer.
             * Then construct the subclass associated with that code,
             * and deserialize the details for that subclass.
             */
            static layer_ptr deserialize(std::istream& in);


            /** \brief Serialize.
             */
            friend inline void
            serialize(const layer& item, std::ostream& out) {
                const int version = 1;
                dlib::serialize(version, out);
                dlib::serialize(item.name, out);
                dlib::serialize(item.inbound, out);
                dlib::serialize(item.code(), out);
                item.serialize_detail(out);
            }

            /** \brief Serialize the layer detail object.
             *
             * This method is used internally by serialize() and
             * net::vector::serialize_native().
             *
             * The default version serializes nothing, and
             * that's ok for layers that don't have any details.
             */
            virtual void serialize_detail(std::ostream&) const {}


            /** \brief Get parameter format for serialize/deserialize.
             *
             * Return the format this layer will be serialized with if
             * pf::native is requested.
             * For objects that were previously deserialized, this method
             * returns the format they were serialized with.
             *
             * The default implementation of this method returns pf::native. 
             * Layers that don't have any parameters that may be subject
             * to alternate format will return this value.
             * Generally, only convolutions and fully connected (fc) layers
             * will override this method to return a more interesting value.
             */
            virtual dlibx::parameter_format parameter_format() const;


            /** \brief Connect layers inbound and outbound.
             *
             * This method verifies the correctness of the vector of layers,
             * and then initializes the fields inbound_nodes and outbound_nodes.
             *
             * To be correct, the layers must have unique non-empty names,
             * the connections must map out a directed acyclic graph, 
             * the first layer must be the only input (with no inbound layers),
             * and the last layer must be the only output.
             * Each layer's inbound names must refer to earlier layers.
             *
             * \returns map of layer name -> layer object
             */
            static std::map<std::string_view, layer*> map_layers(
                stdx::forward_iterator<layer*> first,
                stdx::forward_iterator<layer*> last);

            template <typename ITER>
            friend inline auto map_layers(ITER first, ITER last) {
                const auto make_pointer =
                    [](auto&& x) { return stdx::pointer_to<layer>(x); };
                return layer::map_layers(
                    {first,make_pointer}, {last,make_pointer});
            }


            /** \brief Generate Keras compatible json object.
             */
            static json::object keras(
                stdx::forward_iterator<const layer*> first,
                stdx::forward_iterator<const layer*> last);

            template <typename ITER>
            friend inline auto keras(ITER first, ITER last) {
                const auto make_pointer =
                    [](auto&& x) { return stdx::pointer_to<const layer>(x); };
                return layer::keras({first,make_pointer},{last,make_pointer});
            }


            /** \brief Find common input among multiple inbound nodes.
             *
             * If inbound_nodes.empty(), then nullptr is returned.
             * If inbound_nodes.size() == 1, then inbound_nodes.front()
             * is returned.
             * Otherwise follow each branch defined by inbound_nodes to
             * find a common input node.
             * If stop_at is that input node then return it.
             * However, if stop_at is seen but it is not the common node,
             * then return nullptr.
             */
            const layer* common_input(const layer* stop_at = nullptr) const;


        protected:
            layer() = default;

            /** \brief Construct copy.
             */
            virtual layer_ptr copy_detail() const = 0;

            /** \brief Single Keras compatible object describing layer.
             */
            virtual json::object keras_object() const;

            /** \brief Multiple Keras compatible objects describing layer.
             *
             * The default implementation calls keras_object() and 
             * returns an array with that one object if it's non-empty.
             *
             * Subclasses may override either keras_array() or keras_object().
             */
            virtual json::array keras_array() const;

            /** \brief Forward (not in place).
             *
             * With this method the input tensors are not modified, and
             * the get_output() must be used.
             */
            virtual void
            forward_const(dlib::tensor const* const* inputs,
                          std::size_t num_inputs);

            /** \brief Forward in place.
             *
             * With this method the input tensor may be modifid in place.
             * In this case, output tensor remains unallocated and
             * the input is returned.
             *
             * Otherwise, this method may behave the same as forward_const(), 
             * which is the default implementation.
             */
            virtual dlib::tensor& forward_inplace(dlib::tensor& input);

            /** \brief Tensor from image.
             *
             * The image is converted and stored in output_buffer().
             */
            virtual void to_tensor(
                stdx::forward_iterator<const raw_image::plane&> first,
                stdx::forward_iterator<const raw_image::plane&> last);

            /** \brief Allocate output tensor and return reference.
             */
            inline dlib::resizable_tensor& allocate_output() {
                if (!output_buffer) output_buffer.emplace();
                return *output_buffer;
            }

            /** \brief Deallocate output tensor.
             */
            inline void clear_output() {
                output_buffer = std::nullopt;
                output_tensor = nullptr;
            }


        private:
            /// buffer for input tensors
            std::vector<dlib::tensor const*> input_tensors;

            /// output buffer (not used by layers that operate in place)
            std::optional<dlib::resizable_tensor> output_buffer;

            /// pointer to output from this layer
            dlib::tensor* output_tensor = nullptr;

            /// internal helper
            void concise(std::ostream& out,
                         layer const* stop_at = nullptr) const;

            layer(layer&&) = delete;
            layer(const layer&) = delete;
        };


        inline void serialize(const layer_ptr& item, std::ostream& out) {
            if (!item)
                throw std::invalid_argument("attempt to serialize nullptr");
            serialize(*item, out);
        }
        inline void deserialize(layer_ptr& item, std::istream& in) {
            item = layer::deserialize(in);
        }
    }
}
