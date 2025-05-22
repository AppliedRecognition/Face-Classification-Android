#pragma once

#include <dlib/dnn/layers.h>

namespace dlibx {

    /** \brief Like dlib::prelu_ but with support for per channel scalers.
     */
    class prelu_ {
    public:
        explicit prelu_(float initial_param_value_ = 0.25,
                        bool perchannel = false)
            : initial_param_value(initial_param_value_),
              perchannel(perchannel) {
        }

        prelu_(const dlib::prelu_& other)
            : params(other.get_layer_params()),
              initial_param_value(other.get_initial_param_value()),
              perchannel(false) {
        }

        float get_initial_param_value() const {
            return initial_param_value;
        }

        bool is_per_channel() const {
            return params.size() > 1 || perchannel;
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub) {
            params.set_size(perchannel ? sub.get_output().k() : 1);
            params = initial_param_value;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            const dlib::tensor& input = sub.get_output();
            output.copy_size(input);
            if (params.size() == 1)
                dlib::tt::prelu(output, input, params);
            else {
                DLIB_CASSERT(long(params.size()) == input.k(),
                             "input has incorrect number of channels");
                dlib::alias_tensor el(1);
                dlib::alias_tensor ch(1, 1, input.nr(), input.nc());
                std::size_t chofs = 0;
                for (auto n = input.num_samples(); n > 0; --n)
                    for (long k = 0; k < input.k(); ++k, chofs += ch.size()) {
                        auto src = ch(input, chofs);
                        auto dest = ch(output, chofs);
                        dlib::tt::prelu(dest, src, el(params,size_t(k)));
                    }
            }
        }

        template <typename SUBNET>
        void backward(const dlib::tensor& gradient_input, SUBNET& sub,
                      dlib::tensor& params_grad) {
            if (params.size() == 1)
                dlib::tt::prelu_gradient(sub.get_gradient_input(),
                                         sub.get_output(),
                                         gradient_input, params, params_grad);
            else {
                dlib::tensor& sgin = sub.get_gradient_input();
                dlib::tensor const& sout = sub.get_output();

                DLIB_CASSERT(sgin.size() == sout.size(),
                             "prelu: tensor size mismatch");
                DLIB_CASSERT(sgin.size() == gradient_input.size(),
                             "prelu: tensor size mismatch");
                DLIB_CASSERT(long(params.size()) == sout.k(),
                             "input has incorrect number of channels");
                DLIB_CASSERT(params.size() == params_grad.size(),
                             "input has incorrect number of channels");

                dlib::alias_tensor el(1);
                dlib::alias_tensor ch(1, 1, sout.nr(), sout.nc());
                std::size_t chofs = 0;
                for (auto n = sout.num_samples(); n > 0; --n)
                    for (long k = 0; k < sout.k(); ++k, chofs += ch.size()) {
                        auto ch_sgin = ch(sgin, chofs);
                        auto el_pg = el(params_grad,size_t(k));
                        dlib::tt::prelu_gradient(
                            ch_sgin, ch(sout, chofs),
                            ch(gradient_input, chofs),
                            el(params,size_t(k)), el_pg);
                    }
            }
        }

        inline auto
        map_input_to_output(const dlib::dpoint& p) const { return p; }
        inline auto
        map_output_to_input(const dlib::dpoint& p) const { return p; }

        const dlib::tensor& get_layer_params() const { return params; }
        dlib::tensor& get_layer_params() { return params; }

        friend void serialize(const prelu_& item, std::ostream& out) {
            using dlib::serialize;
            if (item.params.size() <= 1)
                serialize("prelu_", out);
            else
                serialize("prelu_9", out);
            serialize(item.params, out);
            serialize(item.initial_param_value, out);
        }

        friend void deserialize(prelu_& item, std::istream& in) {
            using dlib::deserialize;
            std::string version;
            deserialize(version, in);
            if (version != "prelu_" && version != "prelu_9")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::prelu_.");
            deserialize(item.params, in);
            deserialize(item.initial_param_value, in);
            item.perchannel = item.params.size() > 1;
        }

        friend std::ostream& operator<<(std::ostream& out, const prelu_& item) {
            out << "prelu\t (";
            if (item.params.size() > 1)
                out << "channels=" << item.params.size() << ", ";
            out << "initial_param_value="<<item.initial_param_value
                << ")";
            return out;
        }

        friend void to_xml(const prelu_& item, std::ostream& out) {
            out << "<prelu initial_param_value='"
                << item.initial_param_value <<"'>\n";
            out << mat(item.params);
            out << "</prelu>\n";
        }

    private:
        dlib::resizable_tensor params;
        float initial_param_value;
        bool perchannel;
    };

    template <typename SUBNET>
    using prelu = dlib::add_layer<prelu_, SUBNET>;
}


