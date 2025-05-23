#pragma once

#include <dlib/dnn/core.h>
#include "tensor.hpp"

namespace dlibx {

    /** \brief Same as dlib::extract_ but with runtime dynamic parameters.
     *
     * Serializes the same as dlib::extract_.
     */
    class extract_ {
    public:
        extract_() = default;

        explicit extract_(long offset, long k, long nr = 1, long nc = 1)
            : m_offset(offset), m_k(k), m_nr(nr), m_nc(nc) {
            DLIB_CASSERT(offset >= 0, "The offset must be >= 0.");
            DLIB_CASSERT(k > 0, "The number of channels must be > 0.");
            DLIB_CASSERT(nr > 0, "The number of rows must be > 0.");
            DLIB_CASSERT(nc > 0, "The number of columns must be > 0.");
        }
        
        template <long _offset, long _k, long _nr, long _nc>
        extract_(const dlib::extract_<_offset,_k,_nr,_nc>& other)
            : m_offset(_offset), m_k(_k), m_nr(_nr), m_nc(_nc) {
        }

        inline auto offset() const { return m_offset; }
        inline auto k() const { return m_k; }
        inline auto nr() const { return m_nr; }
        inline auto nc() const { return m_nc; }

        template <typename SUBNET>
        void setup (const SUBNET& sub) {
            DLIB_CASSERT(sub.get_output().size() >= sub.get_output().num_samples()*(m_offset+m_k*m_nr*m_nc), "The tensor we are trying to extract from the input tensor is too big to fit into the input tensor.");
            aout = { sub.get_output().num_samples(), m_k*m_nr*m_nc };
            ain  = { sub.get_output().num_samples(),
                long(sub.get_output().size())/sub.get_output().num_samples() };
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            if (aout.num_samples() != sub.get_output().num_samples()) {
                aout = { sub.get_output().num_samples(), m_k*m_nr*m_nc };
                ain  = { sub.get_output().num_samples(),
                    long(sub.get_output().size())/sub.get_output().num_samples() };
            }
            output.set_size(sub.get_output().num_samples(), m_k, m_nr, m_nc);
            auto out = aout(output,0);
            auto in = ain(sub.get_output(),0);
            dlib::tt::copy_tensor(
                false, out, 0,
                in, std::size_t(m_offset), std::size_t(m_k*m_nr*m_nc));
        } 

        template <typename SUBNET>
        void backward(const dlib::tensor& gradient_input, SUBNET& sub,
                      dlib::tensor& /*params_grad*/) {
            auto out = ain(sub.get_gradient_input(),0);
            auto in = aout(gradient_input,0);
            dlib::tt::copy_tensor(
                true, out, std::size_t(m_offset),
                in, 0, std::size_t(m_k*m_nr*m_nc));
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const extract_& item, std::ostream& out) {
            using dlib::serialize;
            serialize("extract_", out);
            serialize(item.m_offset, out);
            serialize(item.m_k, out);
            serialize(item.m_nr, out);
            serialize(item.m_nc, out);
        }

        friend void deserialize(extract_& item, std::istream& in) {
            using dlib::deserialize;
            std::string version;
            deserialize(version, in);
            if (version != "extract_")
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::extract_.");

            long offset;
            long k;
            long nr;
            long nc;
            deserialize(offset, in);
            deserialize(k, in);
            deserialize(nr, in);
            deserialize(nc, in);

            if (offset < 0 || k < 1 || nr < 1 || nc < 1)
                throw dlib::serialization_error("Invalid parameters found while deserializing dlibx::extract_");
            item.m_k = k;
            item.m_nr = nr;
            item.m_nc = nc;
            item.m_offset = offset;
            item.aout = {};
            item.ain = {};
        }

        friend std::ostream&
        operator<<(std::ostream& out, const extract_& item) {
            out << "extract\t ("
                << "offset="<<item.m_offset
                << ", k="<<item.m_k
                << ", nr="<<item.m_nr
                << ", nc="<<item.m_nc
                << ")";
            return out;
        }

        friend void to_xml(const extract_& item, std::ostream& out) {
            out << "<extract";
            out << " offset='"<<item.m_offset<<"'";
            out << " k='"<<item.m_k<<"'";
            out << " nr='"<<item.m_nr<<"'";
            out << " nc='"<<item.m_nc<<"'";
            out << "/>\n";
        }

    private:
        long m_offset = 0, m_k = 0, m_nr = 0, m_nc = 0;
        dlib::alias_tensor aout, ain;
        dlib::resizable_tensor params; // unused
    };

    template <typename SUBNET>
    using extract = dlib::add_layer<extract_,SUBNET>;
}    
