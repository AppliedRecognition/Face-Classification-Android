#pragma once

#include <dlib/dnn/core.h>
//#include <dlib/dnn/layers.h>
//#include <dlib/dnn/utilities.h>

namespace dlibx {

    using dlib::tensor;

    /** \brief Relu with upper cap at N.
     *
     * This object serializes and deserializes as a standard boundless relu.
     * The value N is not stored.
     */
    template <long N>
    class relun_ {
        static_assert(N > 0);
        
    public:
        relun_() = default;

        template <typename SUBNET>
        void setup (const SUBNET&) {
        }

        void forward_inplace(const tensor& input, tensor& output) {
            output = upperbound(lowerbound(mat(input), 0), N);
        } 

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& ) {
            //tt::relu_gradient(data_grad, computed_output, gradient_input);
            throw std::runtime_error("relun_::backward() not implemented");
        }

        inline auto map_input_to_output(const dlib::dpoint& p) const {
            return p;
        }
        inline auto map_output_to_input(const dlib::dpoint& p) const {
            return p;
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const relun_&, std::ostream& out) {
            dlib::serialize("relu_", out);
        }

        friend void deserialize(relun_&, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "relu_")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlib::relu_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const relun_&) {
            out << "relu" << N;
            return out;
        }

        friend void to_xml(const relun_&, std::ostream& out) {
            out << "<relu" << N << "/>\n";
        }

    private:
        dlib::resizable_tensor params;
    };

    template <long N, typename SUBNET>
    using relun = dlib::add_layer<relun_<N>, SUBNET>;
    template <typename SUBNET>
    using relu6 = relun<6,SUBNET>;
}
