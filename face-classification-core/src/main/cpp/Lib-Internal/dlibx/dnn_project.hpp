#pragma once

#include <dlib/dnn/core.h>
#include "tensor.hpp"

namespace dlibx {

    /** \brief Layer to train camera projection.
     */
    class project_ {

    public:
        project_(float initial_focal_length = 1000)
            : initial_focal_length(initial_focal_length),
              learning_rate_multiplier(1) {
        }

        float get_learning_rate_multiplier() const {
            return learning_rate_multiplier;
        }
        void set_learning_rate_multiplier(float val) {
            learning_rate_multiplier = val;
        }

        const dlib::tensor& get_layer_params() const { return params; }
        dlib::tensor& get_layer_params() { return params; }

        template <typename SUBNET>
        void setup(const SUBNET& sub) {
            const dlib::tensor& input = sub.get_output();
            const auto sample_size = long(input.k()*input.nr()*input.nc());
            DLIB_CASSERT(sample_size % 3 == 0,
                         "sample size must be a multiple of 3");

            params.set_size(sample_size / 3, 3);

            for (auto p = params.host_write_only(),
                     end = p + params.size(); p != end; p += 3) {
                p[0] = p[1] = 0; // principal point offset
                p[2] = initial_focal_length;
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            if (params.size() <= 0)
                setup(sub);

            const dlib::tensor& input = sub.get_output();
            const auto sample_size = long(input.k()*input.nr()*input.nc());
            DLIB_CASSERT(std::size_t(sample_size) == params.size(),
                         "incorrect number of points per sample");

            const auto num_cameras = sample_size / 3;
            output.set_size(input.num_samples(), num_cameras*2);

            auto const* src = input.host();
            auto* dest = output.host_write_only();

            for (auto j = output.num_samples(); 0 < j; --j) {
                auto p = params.host();
                for (auto i = num_cameras; 0 < i; --i,
                         src += 3, dest += 2, p += 3) {
                    dest[0] = p[0] + src[0]*(p[2]/src[2]);
                    dest[1] = p[1] + src[1]*(p[2]/src[2]);
                }
            }
        }

        template <typename SUBNET>
        void backward(const dlib::tensor& gradient_input, SUBNET& sub,
                      dlib::tensor& params_grad) {

            const dlib::tensor& input = sub.get_output();
            DLIB_CASSERT(gradient_input.num_samples() == input.num_samples());
            DLIB_CASSERT(gradient_input.size()*3 == input.size()*2);

            DLIB_CASSERT(params_grad.size() == params.size());
            const auto num_cameras = params.size() / 3;
            params_grad = 0.0f;

            // compute parameter gradient (only if it will be used)
            if (learning_rate_multiplier > 0) {
                auto const* in = input.host();
                auto const* src = gradient_input.host();
                for (auto j = input.num_samples(); 0 < j; --j) {
                    auto* dest = params_grad.host();
                    for (auto i = num_cameras; 0 < i; --i,
                             src += 2, in += 3, dest += 3) {
                        // gcu = SUM gu
                        // gcv = SUM gv
                        // gf = SUM [ gu*X/Z + gv*Y/Z ]
                        dest[0] += src[0];
                        dest[1] += src[1];
                        dest[2] += (src[0]*in[0] + src[1]*in[1]) / in[2];
                    }
                }
            }

            dlib::tensor& gradient_output = sub.get_gradient_input();
            DLIB_CASSERT(gradient_output.num_samples() == input.num_samples());
            DLIB_CASSERT(gradient_output.size() == input.size());

            // compute data gradient output
            auto const* in = input.host();
            auto const* src = gradient_input.host();
            auto* dest = gradient_output.host_write_only();
            for (auto j = gradient_output.num_samples(); 0 < j; --j) {
                auto const* p = params.host();
                for (auto i = num_cameras; 0 < i; --i,
                         src += 2, dest += 3, in += 3, p += 3) {
                    // gX = gu*f/Z
                    // gY = gv*f/Z
                    // gZ = -gu*X*f/(Z^2) -gv*Y*f/(Z^2)
                    dest[0] = src[0]*(p[2]/in[2]);
                    dest[1] = src[1]*(p[2]/in[2]);
                    dest[2] = -(src[0]*in[0] + src[1]*in[1])*(p[2]/in[2]/in[2]);
                }
            }
        }

        friend void serialize(const project_& item, std::ostream& out) {
            using dlib::serialize;
            serialize("project_1", out);
            serialize(item.params, out);
            serialize(item.initial_focal_length, out);
            serialize(item.learning_rate_multiplier, out);
        }

        friend void deserialize(project_& item, std::istream& in) {
            std::string version;
            using dlib::deserialize;
            deserialize(version, in);
            if (version == "project_1") {
                deserialize(item.params, in);
                deserialize(item.initial_focal_length, in);
                deserialize(item.learning_rate_multiplier, in);
            }
            else
                throw dlib::serialization_error(
                    "Unexpected version '" + version +
                    "' found while deserializing project layer.");
        }

        friend std::ostream&
        operator<<(std::ostream& out, const project_& item) {
            out << "project\t ("
                << "init_focal_length="<<item.initial_focal_length
                << ")";
            out << " learning_rate_mult="<<item.learning_rate_multiplier;
            return out;
        }

        friend void to_xml(const project_& item, std::ostream& out) {
            out << "<project"
                << " init_focal_length='"<<item.initial_focal_length<<"'"
                << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                << ">\n";
            out << mat(item.get_layer_params());
            out << "</project>\n";
        }

    private:
        float initial_focal_length;
        float learning_rate_multiplier;
        dlib::resizable_tensor params;
    };

    template <typename SUBNET>
    using project = dlib::add_layer<project_, SUBNET>;
}



