#pragma once

#include "dnn_traits.hpp"
#include <dlib/dnn/core.h>

namespace dlibx {

    using dlib::tensor;

    enum transpose_mode {
        TRANSPOSE_KRC = 0,  // identify (no transpose)
        TRANSPOSE_KCR = 1,
        TRANSPOSE_RKC = 2,
        TRANSPOSE_CKR = 3,
        TRANSPOSE_RCK = 4,
        TRANSPOSE_CRK = 5
    };
    constexpr auto to_string(transpose_mode tm) {
        switch (tm) {
        case TRANSPOSE_KRC: return "KRC";
        case TRANSPOSE_KCR: return "KCR";
        case TRANSPOSE_RKC: return "RKC";
        case TRANSPOSE_CKR: return "CKR";
        case TRANSPOSE_RCK: return "RCK";
        case TRANSPOSE_CRK: return "CRK";
        default: return "UNKNOWN";
        }
    }

    /** \brief Reorder tensor axis and optionally reshape output.
     *
     * The tensor is first transposed as defined by mode.
     * For example, TRANSPOSE_RCK produces transposed tensor with:
     *    t_k  = in_nr
     *    t_nr = in_nc
     *    t_nc = in_k
     *
     * Then the output is reshaped to the specified dimensions.
     * For each dimension, values may be:
     *     > 0  -> specified exact value
     *     = 0  -> copy value from corresponding transposed dimension
     *     < 0  -> compute value to ensure sample size (ie. v = size / x / y)
     * At most one of these dimensions may be < 0.
     * The resulting dimensions are checked for validity.
     * That is, sample_size == out_k * out_nr * out_nc.
     */
    class transpose_ {
    public:
        transpose_(transpose_mode mode = TRANSPOSE_KRC,
                   long out_k = 0, long out_nr = 0, long out_nc = 0)
            : tmode(mode), out_k(out_k), out_nr(out_nr), out_nc(out_nc) {
        }

        inline auto mode() const { return tmode; }
        inline auto k()  const { return out_k;  }
        inline auto nr() const { return out_nr; }
        inline auto nc() const { return out_nc; }

        template <typename SUBNET>
        void setup (const SUBNET&) {}

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            const tensor& input = sub.get_output();
            const auto sample_size = input.k() * input.nr() * input.nc();

            long long t_k, t_nr, t_nc;
            switch (tmode) {
            case TRANSPOSE_KRC:
                t_k  = input.k();
                t_nr = input.nr();
                t_nc = input.nc();
                break;

            case TRANSPOSE_KCR:
                t_k  = input.k();
                t_nr = input.nc();
                t_nc = input.nr();
                break;

            case TRANSPOSE_RKC:
                t_k  = input.nr();
                t_nr = input.k();
                t_nc = input.nc();
                break;

            case TRANSPOSE_CKR:
                t_k  = input.nc();
                t_nr = input.k();
                t_nc = input.nr();
                break;

            case TRANSPOSE_RCK:
                t_k  = input.nr();
                t_nr = input.nc();
                t_nc = input.k();
                break;

            case TRANSPOSE_CRK:
                t_k  = input.nc();
                t_nr = input.nr();
                t_nc = input.k();
                break;

            default:
                throw std::invalid_argument(
                    "invalid mode in transpose::forward()");
            }

            {
                auto nk = out_k  ? out_k  : t_k;
                auto nr = out_nr ? out_nr : t_nr;
                auto nc = out_nc ? out_nc : t_nc;
                if (nk < 0)
                    nk = sample_size / nr / nc;
                else if (nr < 0)
                    nr = sample_size / nk / nc;
                else if (nc < 0)
                    nc = sample_size / nk / nr;
                if (nk <= 0 || nr <= 0 || nc <= 0 || sample_size != nk*nr*nc)
                    std::logic_error("size mismatch in transpose::forward()");
                output.set_size(input.num_samples(), nk, nr, nc);
            }

            auto src = input.host();
            auto dest = output.host_write_only();

            const auto ir = input.nr();
            const auto ic = input.nc();

            switch (tmode) {
            case TRANSPOSE_KRC:
                memcpy(dest, src, output.size() * sizeof(float));
                break;

            case TRANSPOSE_KCR:
                for (auto n = input.num_samples()*input.k(); n > 0; --n) {
                    for (long c = 0; c < ic; ++c)
                        for (long r = 0; r < ir; ++r)
                            *dest++ = src[r*ic + c];
                    src += ir*ic;
                }
                break;

            case TRANSPOSE_RKC:
                for (auto n = output.num_samples(); n > 0; --n) {
                    for (long r = 0; r < ir; ++r)
                        for (long k = 0; k < input.k(); ++k)
                            dest = std::copy_n(src+(k*ir+r)*ic, ic, dest);
                    src += sample_size;
                }
                break;

            case TRANSPOSE_CKR:
                for (auto n = output.num_samples(); n > 0; --n) {
                    for (long c = 0; c < ic; ++c)
                        for (long k = 0; k < input.k(); ++k)
                            for (long r = 0; r < ir; ++r)
                                *dest++ = src[(k*ir+r)*ic + c];
                    src += sample_size;
                }
                break;

            case TRANSPOSE_RCK:
                for (auto n = output.num_samples(); n > 0; --n) {
                    for (long r = 0; r < ir; ++r)
                        for (long c = 0; c < ic; ++c)
                            for (long k = 0; k < input.k(); ++k)
                                *dest++ = src[(k*ir+r)*ic + c];
                    src += sample_size;
                }
                break;

            case TRANSPOSE_CRK:
                for (auto n = output.num_samples(); n > 0; --n) {
                    for (long c = 0; c < ic; ++c)
                        for (long r = 0; r < ir; ++r)
                            for (long k = 0; k < input.k(); ++k)
                                *dest++ = src[(k*ir+r)*ic + c];
                    src += sample_size;
                }
                break;
            }
        }

        template <typename SUBNET>
        void backward(const tensor&, SUBNET& sub, tensor&) {
            if (&sub != &input_layer(sub))
                throw std::runtime_error(
                    "transpose::backward() not implemented");
        }

        inline auto map_input_to_output(dlib::dpoint) const {
            throw std::runtime_error("transpose::map() not implemented");
        }
        inline auto map_output_to_input(dlib::dpoint) const {
            throw std::runtime_error("transpose::map() not implemented");
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const transpose_&item, std::ostream& out) {
            dlib::serialize("transpose_", out);
            dlib::serialize(item.tmode, out);
            dlib::serialize(item.out_k, out);
            dlib::serialize(item.out_nr, out);
            dlib::serialize(item.out_nc, out);
        }

        friend void deserialize(transpose_& item, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "transpose_")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::transpose_.");
            int mode;
            dlib::deserialize(mode, in);
            if (mode < 0 || 5 < mode)
                throw dlib::serialization_error("Invalid mode found while deserializing dlibx::transpose_.");
            item.tmode = transpose_mode(mode);
            dlib::deserialize(item.out_k, in);
            dlib::deserialize(item.out_nr, in);
            dlib::deserialize(item.out_nc, in);
        }

        friend std::ostream&
        operator<<(std::ostream& out, const transpose_& item) {
            out << "transpose ("
                << to_string(item.tmode)
                << ", k=" << item.out_k
                << ", nr=" << item.out_nr
                << ", nc=" << item.out_nc
                << ')';
            return out;
        }

        friend void to_xml(const transpose_& item, std::ostream& out) {
            out << "<transpose"
                << " mode='" << to_string(item.tmode) << "'"
                << " k='" << item.out_k << "'"
                << " nr='" << item.out_nr << "'"
                << " nc='" << item.out_nc << "'"
                << "/>\n";
        }

    private:
        dlib::resizable_tensor params;  // always empty

        transpose_mode tmode;
        long out_k, out_nr, out_nc;
    };

    /*
    template <long k, long nr, long nc, typename SUBNET>
    using transpose_kcr =
                   dlib::add_layer<transpose_<TRANSPOSE_KCR,k,nr,nc>, SUBNET>;
    template <long k, long nr, long nc, typename SUBNET>
    using transpose_rkc =
                   dlib::add_layer<transpose_<TRANSPOSE_RKC,k,nr,nc>, SUBNET>;
    template <long k, long nr, long nc, typename SUBNET>
    using transpose_ckr =
                   dlib::add_layer<transpose_<TRANSPOSE_CKR,k,nr,nc>, SUBNET>;
    template <long k, long nr, long nc, typename SUBNET>
    using transpose_rck =
                   dlib::add_layer<transpose_<TRANSPOSE_RCK,k,nr,nc>, SUBNET>;
    template <long k, long nr, long nc, typename SUBNET>
    using transpose_crk =
                   dlib::add_layer<transpose_<TRANSPOSE_CRK,k,nr,nc>, SUBNET>;
    */
}
