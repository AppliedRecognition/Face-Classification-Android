#pragma once

#include <dlib/dnn/utilities.h>  // for log1pexp() needed by loss.h
#include <dlib/dnn/loss.h>

namespace dlibx {

    /** \brief Arcface loss function.
     */
    class loss_arcface_ {
    public:
        using training_label_type = unsigned long;
        using output_label_type = unsigned long;

        loss_arcface_(float margin = 0.5, float scale = 64)
            : m_margin(margin), m_scale(scale) {
            DLIB_CASSERT(0 <= m_margin);
            DLIB_CASSERT(0 < m_scale);
        }

        float get_margin() const {
            return m_margin;
        }

        float get_scale() const {
            return m_scale;
        }

        template <typename SUB_TYPE, typename label_iterator>
        void to_label(const dlib::tensor& input, const SUB_TYPE& sub,
                      label_iterator iter) const {

            const dlib::tensor& output = sub.get_output();
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(output.nr() == 1 && output.nc() == 1 );
            DLIB_CASSERT(input.num_samples() == output.num_samples());

            // Note that output.k() should match the number of labels.
            // The index of the largest output for this sample is the label.
            for (long i = 0; i < output.num_samples(); ++i)
                *iter++ = index_of_max(rowm(mat(output),i));
        }

        template <typename const_label_iterator, typename SUBNET>
        double compute_loss_value_and_gradient(
            const dlib::tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub) const {

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);

            // input_tensor is the input to the entire net
            const auto num_samples = input_tensor.num_samples();
            DLIB_CASSERT(0 < num_samples);

            // output is the computed class vector
            const dlib::tensor& output = sub.get_output();
            DLIB_CASSERT(output.num_samples() == num_samples);
            DLIB_CASSERT(output.nr() == 1 && output.nc() == 1);
            const auto num_classes = output.k();

            dlib::tensor& grad = sub.get_gradient_input();
            DLIB_CASSERT(grad.num_samples() == num_samples);
            DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);
            DLIB_CASSERT(grad.k() == num_classes);

            // use grad a temporary
            memcpy(grad, output);

            // The network must produce a number of outputs that is
            // equal to the number of labels when using this type of loss.

            // add margin to true index and scale all values
            {
                auto t = truth;
                float* g = grad.host();
                for (auto n = num_samples; n > 0; --n, ++t) {
                    const auto y = long(*t);
                    DLIB_CASSERT(y < num_classes);
                    auto angle = g[y] >= 1 ? 0 :
                        g[y] <= -1 ? M_PI : std::acos(g[y]);
                    angle += m_margin;
                    g[y] = angle < M_PI ? std::cos(angle) : -1;
                    for (auto k = num_classes; k > 0; --k, ++g)
                        *g *= m_scale;
                }
            }
            
            dlib::tt::softmax(grad, grad);

            // loss is the average loss over the mini-batch
            const auto scale = 1.0 / double(num_samples);
            double loss = 0;
            float* g = grad.host();
            for (auto n = num_samples; n > 0; --n, ++truth) {
                const auto y = long(*truth);
                for (long k = 0; k < num_classes; ++k, ++g) {
                    if (k == y) {
                        loss += scale * -dlib::safe_log(*g);
                        *g = scale * (*g - 1);
                    }
                    else
                        *g = scale * *g;
                }
            }
            return loss;
        }

        friend void serialize(const loss_arcface_& item, std::ostream& out) {
            using dlib::serialize;
            serialize("loss_arcface_", out);
            serialize(item.m_margin, out);
            serialize(item.m_scale, out);
        }

        friend void deserialize(loss_arcface_& item, std::istream& in) {
            using dlib::deserialize;
            using dlib::serialization_error;
            std::string version;
            deserialize(version, in);
            if (version != "loss_arcface_")
                throw serialization_error("Unexpected version found while deserializing dlibx::loss_arcface_.");
            deserialize(item.m_margin, in);
            deserialize(item.m_scale, in);
            if (!(0 <= item.m_margin))
                throw serialization_error("Invalid margin found while deserializing dlibx::loss_arcface_.");
            if (!(0 < item.m_scale))
                throw serialization_error("Invalid scale found while deserializing dlibx::loss_arcface_.");
        }

        friend std::ostream&
        operator<<(std::ostream& out, const loss_arcface_& item) {
            out << "loss_arcfacet ("
                << " margin=" << item.m_margin
                << ", scale=" << item.m_scale
                << ")";
            return out;
        }

        friend void to_xml(const loss_arcface_& item, std::ostream& out) {
            out << "<loss_arcface"
                << " margin='" << item.m_margin << "'"
                << " scale='" << item.m_scale << "'"
                << "/>";
        }

    private:
        float m_margin, m_scale;
    };

    template <typename SUBNET>
    using loss_arcface = dlib::add_loss_layer<loss_arcface_, SUBNET>;

}
