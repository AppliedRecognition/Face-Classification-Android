#pragma once

#include <dlib/dnn/core.h>


namespace dlibx {

    using dlib::tensor;

    /** \brief Sum neighbouring channels.
     *
     * This layer is useful to create a SpatialCrossMapLRN.
     * Specifically do: 
     *     mult_prev<lambda<sum_neighbours<size,lambda<tag<input>>>>>.
     * Where first (left) lamba is y = pow(k+(alpha/size)*x, -beta), and
     * second (right) lambda is y = x*x.
     */
    template <long SIZE>
    class sum_neighbours_ {
        static_assert(SIZE > 0 && (SIZE&1) == 1, "SIZE must be an odd number");

    public:
        sum_neighbours_() = default;

        template <typename SUBNET>
        void setup (const SUBNET&) {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            const tensor& data = sub.get_output();
            const auto channel_size = data.nr() * data.nc();
            const auto memadd =
                [channel_size](auto* dest, const auto* src) {
                    for (auto j = channel_size; j > 0; --j, ++dest, ++src)
                        *dest += *src;
                };
            output.copy_size(data);
            auto src = data.host();
            auto dest = output.host_write_only();
            for (auto n = data.num_samples(); n > 0; --n) {
                for (long k = 0; k < data.k(); ++k) {
                    memcpy(dest, src, std::size_t(channel_size)*sizeof(float));
                    for (auto ofs = std::min(SIZE/2, k); ofs > 0; --ofs)
                        memadd(dest, src - ofs*channel_size);
                    for (auto ofs = std::min(SIZE/2, long(data.k()-1-k)); ofs > 0; --ofs)
                        memadd(dest, src + ofs*channel_size);
                    dest += channel_size;
                    src += channel_size;
                }
            }
        } 

        template <typename SUBNET>
        void backward(const tensor&, SUBNET&, tensor&) {
            throw std::runtime_error("sum_neighbours_::backward() not implemented");
        }

        inline auto map_input_to_output(const dlib::dpoint& p) const {
            return p;
        }
        inline auto map_output_to_input(const dlib::dpoint& p) const {
            return p;
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const sum_neighbours_&, std::ostream& out) {
            dlib::serialize("sum_neighbours_", out);
        }

        friend void deserialize(sum_neighbours_&, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "sum_neighbours_")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::sum_neighbours_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const sum_neighbours_&) {
            out << "sum_neighbours";
            return out;
        }

        friend void to_xml(const sum_neighbours_&, std::ostream& out) {
            out << "<sum_neighbours" << "/>\n";
        }

    private:
        dlib::resizable_tensor params;
    };

    template <long size, typename SUBNET>
    using sum_neighbours = dlib::add_layer<sum_neighbours_<size>, SUBNET>;
}
