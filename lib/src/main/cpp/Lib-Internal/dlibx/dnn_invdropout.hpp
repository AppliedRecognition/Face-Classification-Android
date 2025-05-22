#pragma once

#include "float_constants.hpp"
#include <dlib/dnn/core.h>

namespace dlibx {

    /** \brief Inverted dropout
     *
     * Define keep_rate as 1 - drop_rate. 
     *
     * This layer zeros out a randomly selected drop_rate fraction of
     * the values passing through, while also scaling the non-zeroed
     * values by 1 / keep_rate.
     *
     * This layer will only operator on batches of more than one sample.
     * In the case of a single sample passing through it is assumed this
     * is a test or usage sample, and it will not be modified.
     */
    template <typename INIT = float_half>
    class invdropout_ {
    public:
        invdropout_(float drop_rate = INIT{})
            : drop_rate(drop_rate),
              scale(1 / (1-drop_rate)),
              rnd(std::rand()) {
            DLIB_CASSERT(0 <= drop_rate && drop_rate < 1);
        }

        // the rnd object is non-copyable
        // and we don't care what the template parameter is
        template <typename OTHER>
        invdropout_(const invdropout_<OTHER>& other)
            : drop_rate(other.drop_rate),
              scale(1 / (1-drop_rate)),
              mask(other.mask),
              rnd(std::rand()) {}
        invdropout_(const invdropout_& other)
            : drop_rate(other.drop_rate),
              scale(1 / (1-drop_rate)),
              mask(other.mask),
              rnd(std::rand()) {}
        invdropout_& operator=(const invdropout_& other) {
            if (this != &other) {
                drop_rate = other.drop_rate;
                scale = 1 / (1-drop_rate);
                mask = other.mask;
            }
            return *this;
        }

        constexpr float get_drop_rate() const { return drop_rate; }

        template <typename SUBNET>
        void setup(const SUBNET&) {}

        void forward_inplace(const dlib::tensor& input, dlib::tensor& output) {
            if (input.num_samples() > 1) {
                // create a random mask and use it to filter the data
                mask.copy_size(input);
                rnd.fill_uniform(mask);
                using namespace dlib::tt;
                threshold(mask, drop_rate);
                affine_transform(mask, mask, scale);
                multiply(false, output, input, mask);
            }
        } 

        void backward_inplace(
            const dlib::tensor& gradient_input, dlib::tensor& data_grad,
            dlib::tensor&) {
            if (gradient_input.num_samples() > 1)
                dlib::tt::multiply(
                    !is_same_object(gradient_input, data_grad),
                    data_grad, mask, gradient_input);
        }

        inline auto map_input_to_output(const dlib::dpoint& p) const {
            return p;
        }
        inline auto map_output_to_input(const dlib::dpoint& p) const {
            return p;
        }

        const dlib::tensor& get_layer_params() const { return params; }
        dlib::tensor& get_layer_params() { return params; }

        friend void serialize(const invdropout_& item, std::ostream& out) {
            using dlib::serialize;
            serialize("invdropout_", out);
            serialize(item.drop_rate, out);
            serialize(item.mask, out);
        }

        friend void deserialize(invdropout_& item, std::istream& in) {
            using dlib::deserialize;
            std::string version;
            deserialize(version, in);
            if (version != "invdropout_")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::invdropout_.");
            deserialize(item.drop_rate, in);
            item.scale = 1 / (1 - item.drop_rate);
            deserialize(item.mask, in);
        }

        void clean() {
            mask.clear();
        }

        friend std::ostream& operator<<(
            std::ostream& out, const invdropout_& item) {
            out << "invdropout\t ("
                << "drop_rate=" << item.drop_rate
                << ")";
            return out;
        }

        friend void to_xml(const invdropout_& item, std::ostream& out) {
            out << "<invdropout"
                << " drop_rate='" << item.drop_rate << "'";
            out << "/>\n";
        }

    private:
        float drop_rate, scale;
        dlib::resizable_tensor mask;

        dlib::tt::tensor_rand rnd;
        dlib::resizable_tensor params; // unused
    };


    template <typename SUBNET>
    using invdropout = dlib::add_layer<invdropout_<>, SUBNET>;
}
