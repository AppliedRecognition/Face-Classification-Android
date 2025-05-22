#pragma once

#include <dlib/dnn/core.h>
#include "tensor.hpp"

namespace dlibx {

    /** \brief Like dlib::add_prev but crop to size of smaller input.
     */
    template <template<typename> class tag>
    class add_cropped_ {
    public:
        const static unsigned long id = dlib::tag_id<tag>::id;

        add_cropped_() = default;

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/) {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            auto&& t1 = sub.get_output();
            auto&& t2 = dlib::layer<tag>(sub).get_output();
            output.set_size(std::min(t1.num_samples(),t2.num_samples()),
                            std::min(t1.k(),t2.k()),
                            std::min(t1.nr(),t2.nr()),
                            std::min(t1.nc(),t2.nc()));
            dlib::tt::add(output, t1, t2);
        }

        template <typename SUBNET>
        void backward(const dlib::tensor& input, SUBNET& sub, dlib::tensor&) {
            // The gradient just flows backwards to the two layers that
            // forward() added together.
            dlib::tt::add(sub.get_gradient_input(),
                          sub.get_gradient_input(), input);
            dlib::tt::add(dlib::layer<tag>(sub).get_gradient_input(),
                          dlib::layer<tag>(sub).get_gradient_input(), input);
        }

        const dlib::tensor& get_layer_params() const { return params; }
        dlib::tensor& get_layer_params() { return params; }

        inline auto map_input_to_output(const dlib::dpoint& p) const {
            return p;
        }
        inline auto map_output_to_input(const dlib::dpoint& p) const {
            return p;
        }

        friend void serialize(const add_cropped_& /*item*/, std::ostream& out) {
            dlib::serialize("add_cropped_", out);
        }

        friend void deserialize(add_cropped_& /*item*/, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "add_cropped_")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlib::add_cropped_.");
        }
        friend std::ostream& operator<<(std::ostream& out, const add_cropped_& /*item*/) {
            out << "add_cropped"<<id;
            return out;
        }

        friend void to_xml(const add_cropped_& /*item*/, std::ostream& out) {
            out << "<add_cropped tag='"<<id<<"'/>\n";
        }

    private:
        dlib::resizable_tensor params;
    };

    template <template<typename> class tag, typename SUBNET>
    using add_cropped = dlib::add_layer<add_cropped_<tag>, SUBNET>;

    template <typename SUBNET> using add_cropped1  = add_cropped<dlib::tag1, SUBNET>;
    template <typename SUBNET> using add_cropped2  = add_cropped<dlib::tag2, SUBNET>;
    template <typename SUBNET> using add_cropped3  = add_cropped<dlib::tag3, SUBNET>;
    template <typename SUBNET> using add_cropped4  = add_cropped<dlib::tag4, SUBNET>;
    template <typename SUBNET> using add_cropped5  = add_cropped<dlib::tag5, SUBNET>;
    template <typename SUBNET> using add_cropped6  = add_cropped<dlib::tag6, SUBNET>;
    template <typename SUBNET> using add_cropped7  = add_cropped<dlib::tag7, SUBNET>;
    template <typename SUBNET> using add_cropped8  = add_cropped<dlib::tag8, SUBNET>;
    template <typename SUBNET> using add_cropped9  = add_cropped<dlib::tag9, SUBNET>;
    template <typename SUBNET> using add_cropped10 = add_cropped<dlib::tag10, SUBNET>;

    using add_cropped1_  = add_cropped_<dlib::tag1>;
    using add_cropped2_  = add_cropped_<dlib::tag2>;
    using add_cropped3_  = add_cropped_<dlib::tag3>;
    using add_cropped4_  = add_cropped_<dlib::tag4>;
    using add_cropped5_  = add_cropped_<dlib::tag5>;
    using add_cropped6_  = add_cropped_<dlib::tag6>;
    using add_cropped7_  = add_cropped_<dlib::tag7>;
    using add_cropped8_  = add_cropped_<dlib::tag8>;
    using add_cropped9_  = add_cropped_<dlib::tag9>;
    using add_cropped10_ = add_cropped_<dlib::tag10>;
}


