#pragma once

#include <dlib/dnn/core.h>
#include "conv.hpp"

namespace dlibx {

    using dlib::tensor;

    /** \brief Zero padding.
     */
    template <long top, long bottom = top,
              long left = top, long right = bottom>
    class padding_ {
        static_assert(top >= 0 && bottom >= 0 && left >= 0 && right >= 0);

    public:
        padding_() = default;

        template <typename SUBNET>
        void setup (const SUBNET&) {
        }

        template <typename SUBNET>
        inline void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            apply_padding(sub.get_output(), output, top, left, bottom, right);
        } 

        template <typename SUBNET>
        void backward(const tensor&, SUBNET&, tensor&) {
            throw std::runtime_error("padding_::backward() not implemented");
        }

        inline auto map_input_to_output(dlib::dpoint p) const {
            p.x() += left;
            p.y() += top;
            return p;
        }
        inline auto map_output_to_input(dlib::dpoint p) const {
            p.x() -= left;
            p.y() -= top;
            return p;
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const padding_&, std::ostream& out) {
            dlib::serialize("padding_", out);
        }

        friend void deserialize(padding_&, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "padding_")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::padding_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const padding_&) {
            out << "padding";
            return out;
        }

        friend void to_xml(const padding_&, std::ostream& out) {
            out << "<padding" << "/>\n";
        }

    private:
        dlib::resizable_tensor params;
    };

    template <long top, long bottom, typename SUBNET>
    using padding = dlib::add_layer<padding_<top,bottom>, SUBNET>;
}
