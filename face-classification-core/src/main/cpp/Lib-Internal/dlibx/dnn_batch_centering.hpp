#pragma once

#include <dlib/dnn/core.h>
#include <dlib/dnn/layers.h>
//#include "tensor.hpp"


namespace dlibx {
    class batch_centering_ {
    public:
        explicit batch_centering_(unsigned long window_size)
            : num_updates(0), 
              running_stats_window_size(window_size) {
            DLIB_CASSERT(window_size > 0, "The batch centering running stats window size can't be 0.");
        }

        batch_centering_() : batch_centering_(100) {}

        unsigned long get_running_stats_window_size() const {
            return running_stats_window_size;
        }
        void set_running_stats_window_size(unsigned long new_window_size) {
            DLIB_CASSERT(new_window_size > 0, "The batch centering running stats window size can't be 0.");
            running_stats_window_size = new_window_size; 
        }

        inline auto map_input_to_output(const dlib::dpoint& p) const {
            return p;
        }
        inline auto map_output_to_input(const dlib::dpoint& p) const {
            return p;
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub) {
            const dlib::tensor& input = sub.get_output();
            means.set_size(1, input.k(), input.nr(), input.nc());
            running_means.copy_size(means);
            running_means = 0.0f;
            num_updates = 0;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            const dlib::tensor& input = sub.get_output();

            output.copy_size(input);
            const auto num = input.k() * input.nr() * input.nc();
            
            if (input.num_samples() > 1) {
                const double decay = 1.0 - num_updates/(num_updates+1.0);
                ++num_updates;
                if (num_updates > running_stats_window_size)
                    num_updates = running_stats_window_size;

                means.set_size(1, input.k(), input.nr(), input.nc());
                running_means.copy_size(means);

                // compute means
                means = 0;
                const auto p_means = means.host();
                auto src = input.host();
                for (long i = 0; i < num; ++i)
                    for (long n = 0; n < input.num_samples(); ++n)
                        p_means[i] += src[n*num+i];
                means /= input.num_samples();
                means.host(); // copy data back to host

                // subtract means
                src = input.host();
                auto dest = output.host();
                for (long n = 0; n < input.num_samples(); ++n)
                    for (long i = 0; i < num; ++i, ++src, ++dest)
                        *dest = *src - p_means[i];

                // keep track of running average mean
                if (decay < 1)
                    running_means =
                        (1-decay)*mat(running_means) + decay*mat(means);
                else
                    running_means = means;
            }

            else if (running_means.size() > 0) {
                // inference mode -- just subtract mean
                auto src = input.host();
                auto dest = output.host();
                const float* m = running_means.host();
                for (long n = 0; n < input.num_samples(); ++n)
                    for (long k = 0; k < num; ++k, ++dest, ++src)
                        *dest = *src - m[k];
            }

            else // we haven't processed any batches yet so just copy input
                memcpy(output, input);
        }

        template <typename SUBNET>
        void backward(const dlib::tensor& gradient_input,
                      SUBNET& sub, dlib::tensor&) {
            dlib::tt::add(sub.get_gradient_input(),
                          sub.get_gradient_input(), gradient_input);
        }

        explicit operator dlib::affine_() const {
            dlib::affine_ a;
            if (running_means.size() > 0) {
                // have serialize and then deserialize to construct affine
                auto alias =
                    dlib::alias_tensor(1, running_means.k(),
                                       running_means.nr(),
                                       running_means.nc());
                dlib::resizable_tensor t(2 * long(alias.size()));
                alias(t,0) = 1.0f;
                alias(t,alias.size()) = -mat(running_means);
                using dlib::serialize;
                std::stringstream strm;
                serialize("affine_", strm);
                serialize(t, strm);
                serialize(alias, strm);
                serialize(alias, strm);
                serialize(int(dlib::FC_MODE), strm);
                deserialize(a, strm);
            }
            return a;
        }

        const dlib::tensor& get_layer_params() const { return params; }
        dlib::tensor& get_layer_params() { return params; }

        friend void serialize(const batch_centering_& item, std::ostream& out) {
            using dlib::serialize;
            serialize("batch_centering", out);
            serialize(item.running_stats_window_size, out);
            serialize(item.num_updates, out);
            serialize(item.running_means, out);
            serialize(item.means, out);
        }

        friend void deserialize(batch_centering_& item, std::istream& in) {
            using dlib::deserialize;
            std::string version;
            deserialize(version, in);
            if (version != "batch_centering")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::batch_centering_.");
            deserialize(item.running_stats_window_size, in);
            deserialize(item.num_updates, in);
            deserialize(item.running_means, in);
            deserialize(item.means, in);
        }

        friend std::ostream&
        operator<<(std::ostream& out, const batch_centering_& item) {
            out << "batch_centering running_stats_window_size="
                << item.running_stats_window_size;
            return out;
        }

        friend void to_xml(const batch_centering_& item, std::ostream& out) {
            out << "<batch_centering";
            out << " running_stats_window_size='"
                << item.running_stats_window_size << "'";
            out << "/>\n";
        }


    private:
        dlib::resizable_tensor params; // empty
        dlib::resizable_tensor means, running_means;

        unsigned long num_updates;
        unsigned long running_stats_window_size;
    };

    template <typename SUBNET>
    using batch_centering = dlib::add_layer<batch_centering_, SUBNET>;
}
