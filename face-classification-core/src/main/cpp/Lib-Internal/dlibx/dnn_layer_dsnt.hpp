#pragma once

#include <dlib/dnn/layers.h>

namespace dlibx {

    /** \brief Differentiable Spacial to Numerical Transform
     *
     * Output is 1x1 with 2*k channels, where input has k channels.
     * For each input channel, (x,y) outputs are computed.
     * Input must be non-negative.
     */
    class dsnt_ {
    public:
        dsnt_() = default;

        template <typename SUBNET>
        inline void setup(const SUBNET&) {}
        
        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& out) {
            const auto& in = sub.get_output();
            if (long(x_table.size()) != in.nc())
                create_table(x_table, in.nc());
            if (long(y_table.size()) != in.nr())
                create_table(y_table, in.nr());
            out.set_size(in.num_samples(), 2*in.k(), 1, 1);
            auto dest = out.host();
            auto src = in.host();
            for (auto n = in.num_samples()*in.k(); n > 0; --n) {
                float x = 0, y = 0; //, t = 0;
                for (long j = 0; j < in.nr(); ++j) {
                    const auto y_coeff = y_table[std::size_t(j)];
                    auto x_coeff = x_table.begin();
                    for (long i = 0; i < in.nc(); ++i, ++src, ++x_coeff) {
                        //t += *src;
                        y += *src * y_coeff;
                        x += *src * *x_coeff;
                    }
                }
                //if (t > 0) x /= t, y /= t;
                *dest++ = x;
                *dest++ = y;
            }
        }

        template <typename SUBNET>
        void backward(
            const dlib::tensor& computed_output,  // optional
            const dlib::tensor& gradient_input,
            SUBNET& sub,
            dlib::tensor& /*params_grad*/) {

            DLIB_CASSERT(have_same_dimensions(computed_output, gradient_input));
            DLIB_CASSERT(gradient_input.nc() == 1 && gradient_input.nr() == 1);
            
            auto& grad = sub.get_gradient_input();

            DLIB_CASSERT(grad.nc() == long(x_table.size()) &&
                         grad.nr() == long(y_table.size()));
            DLIB_CASSERT(2*grad.k() == gradient_input.k());
            DLIB_CASSERT(grad.num_samples() == gradient_input.num_samples());

            //FILE_LOG(logINFO) << "backward: " << t.k() << 'x' << t.nr() << 'x' << t.nc();

            auto dest = grad.host();
            auto gi = gradient_input.host();
            //auto co = computed_output.host();
            
            for (auto n = grad.num_samples()*grad.k(); n > 0; --n, gi += 2
                     /*, co += 2*/) {
                const auto gx = gi[0], gy = gi[1];
                //const auto cx = co[0], cy = co[1];
                //FILE_LOG(logINFO) << "grad:\t" << gx << '\t' << gy;
                for (long j = 0; j < grad.nr(); ++j) {
                    const auto dy = gy * y_table[std::size_t(j)];
                    auto xi = x_table.begin();
                    for (long i = 0; i < grad.nc(); ++i, ++dest) {
                        // derivative of cx*gx + cy*gy wrt pixel z
                        //   cx = ... + x_table[i]*z + ...
                        //   cy = ... + y_table[j]*z + ...
                        const auto dx = gx * *xi++;
                        *dest += dx + dy;
                    }
                }
            }
        }

        inline auto map_input_to_output(const dlib::dpoint& p) const {
            return p;
        }
        inline auto map_output_to_input(const dlib::dpoint& p) const {
            return p;
        }

        const auto& get_layer_params() const { return params; }
        auto& get_layer_params() { return params; }

        friend auto& operator<<(std::ostream& out, const dsnt_&) {
            return out << "dsnt";
        }

        friend void to_xml(const dsnt_&, std::ostream& out) {
            out << "<dsnt/>\n";
        }

        friend void serialize(const dsnt_&, std::ostream& out) {
            dlib::serialize("dsnt_", out);
        }

        friend void deserialize(dsnt_&, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "dsnt_")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::dsnt_.");
        }

    private:
        dlib::resizable_tensor params;
        std::vector<float> x_table, y_table;

        static void create_table(std::vector<float>& table, long n) {
            DLIB_CASSERT(n >= 0);
            table.clear();
            table.reserve(std::size_t(n));
            for (long i = 0; i < n; ++i)
                table.push_back(float(2*i-(n-1))/float(n));
        }
    };

    template <typename SUBNET>
    using dsnt = dlib::add_layer<dsnt_, SUBNET>;
}
