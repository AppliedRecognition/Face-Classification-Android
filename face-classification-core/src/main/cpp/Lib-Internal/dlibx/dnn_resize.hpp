#pragma once

#include <dlib/dnn/core.h>
#include "tensor.hpp"

namespace dlib {
    template <long,long> class resize_to_;
}
namespace dlibx {

    /** \brief Same as dlib::resize_to_<NR,NC> but runtime dynamic.
     *
     * Serialize format is compatible with dlib::resize_to_<NR,NC>.
     */
    class resize_ {
    public:
        resize_(long nr = 0, long nc = 0) : _nr(nr), _nc(nc) {}
        
        template <long NR, long NC>
        resize_(const dlib::resize_to_<NR,NC>&)
            : _nr(NR), _nc(NC) {}

        inline auto nr() const { return _nr; }
        inline auto nc() const { return _nc; }
        
        template <typename SUBNET>
        inline void setup(const SUBNET&) {}
    
        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            DLIB_CASSERT(_nr > 0 && _nc > 0, "dlibx::resize_ not configured");
            auto& input = sub.get_output();
            scale_y = double(_nr) / double(input.nr());
            scale_x = double(_nc) / double(input.nc());
            output.set_size(input.num_samples(), input.k(), _nr, _nc);
            dlib::tt::resize_bilinear(output, sub.get_output());
        }

        template <typename SUBNET>
        void backward(const dlib::tensor& gradient_input, SUBNET& sub,
                      dlib::tensor&) {
            dlib::tt::resize_bilinear_gradient(
                sub.get_gradient_input(), gradient_input);
        }
        
        inline auto map_input_to_output(dlib::dpoint p) const {
            p.x() = p.x() * scale_x;
            p.y() = p.y() * scale_y;
            return p; 
        }
        inline auto map_output_to_input(dlib::dpoint p) const {
            p.x() = p.x() / scale_x;
            p.y() = p.y() / scale_y;
            return p; 
        }

        inline const dlib::tensor& get_layer_params() const { return params; }
        inline dlib::tensor& get_layer_params() { return params; }
        
        friend void serialize(const resize_& item, std::ostream& out) {
            if (item._nr <= 0 || item._nc <= 0)
                throw dlib::serialization_error(
                    "Object dlibx::resize_ not configured"
                    " -- cannot serialize.");
            using dlib::serialize;
            serialize("resize_to_", out);
            serialize(item._nr, out);
            serialize(item._nc, out);
            serialize(item.scale_y, out);
            serialize(item.scale_x, out);
        }
        
        friend void deserialize(resize_& item, std::istream& in) {
            using dlib::deserialize;
            std::string version;
            deserialize(version, in);
            if (version != "resize_to_")
                throw dlib::serialization_error(
                    "Unexpected version '" + version +
                    "' found while deserializing dlibx::resize_.");
            
            deserialize(item._nr, in);
            deserialize(item._nc, in);
            deserialize(item.scale_y, in);
            deserialize(item.scale_x, in);
        }

        friend auto& operator<<(std::ostream& out, const resize_& item) {
            if (item._nr > 0 && item._nc > 0)
                out << "resize_to ("
                    << "nr=" << item._nr
                    << ", nc=" << item._nc
                    << ")";
            else
                out << "resize_to (unknown)";
            return out;
        }
        
        friend void to_xml(const resize_& item, std::ostream& out) {
            out << "<resize_to";
            if (item._nr > 0 && item._nc > 0)
                out << " nr='" << item._nr << "'"
                    << " nc='" << item._nc << "'";
            out << "/>\n";
        }

    private:
        dlib::resizable_tensor params; // empty
        long _nr = 0, _nc = 0;
        double scale_y = 1;
        double scale_x = 1;
    };

    template <typename SUBNET>
    using resize = dlib::add_layer<resize_, SUBNET>;
}
