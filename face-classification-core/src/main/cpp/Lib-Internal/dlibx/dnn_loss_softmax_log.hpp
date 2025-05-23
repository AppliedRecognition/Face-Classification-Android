#pragma once

#include <dlib/dnn/utilities.h>  // for log1pexp() needed by loss.h
#include <dlib/dnn/loss.h>

namespace dlibx {

    /** \brief Like multiclass log, but labels are probability vector
     * instead of class number.
     */
    class loss_softmax_log_ {
    public:
        using training_label_type = dlib::matrix<float,0,1>;
        using output_label_type = dlib::matrix<float,0,1>;

        template <typename SUB_TYPE, typename label_iterator>
        void to_label(const dlib::tensor& input, const SUB_TYPE& sub,
                      label_iterator iter) const {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            const auto& out = sub.get_output();
            DLIB_CASSERT(out.nr() == 1 && out.nc() == 1);
            DLIB_CASSERT(out.num_samples() == input.num_samples());
            dlib::resizable_tensor smax;
            smax.copy_size(out);
            dlib::tt::softmax(smax,out);
            const float* dest = smax.host();
            for (long i = 0; i < smax.num_samples(); ++i, dest += smax.k())
                //*iter++ = trans(rowm(mat(out),i));
                *iter++ = dlib::mat(dest, smax.k(), 1);
        }

        template <typename const_label_iterator, typename SUBNET>
        double compute_loss_value_and_gradient(
            const dlib::tensor& input,
            const_label_iterator truth,
            SUBNET& sub) const {

            const auto& out = sub.get_output();
            auto& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input.num_samples() != 0);
            DLIB_CASSERT(input.num_samples()%sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input.num_samples() == grad.num_samples());
            DLIB_CASSERT(input.num_samples() == out.num_samples());

            DLIB_CASSERT(out.nr() == 1 && out.nc() == 1);
            DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);
            DLIB_CASSERT(grad.k() == out.k());

            dlib::tt::softmax(grad, out); // grad contains computed softmax

            // loss is the average loss over the mini-batch
            const auto scale = float(1.0 / grad.num_samples());
            double loss = 0;
            float* g = grad.host();
            dlib::matrix<float> ytruth;
            for (long i = 0; i < grad.num_samples(); ++i) {
                ytruth = *truth++;
                auto* y = &ytruth(0);
                DLIB_CASSERT(ytruth.nr() == grad.k() && ytruth.nc() == 1);
                for (long k = 0; k < grad.k(); ++k, ++g, ++y) {
                    if (*y > 0) 
                        loss -= *y * dlib::safe_log(*g, 1e-10f);
                    *g = scale * (*g - *y);
                }
            }
            DLIB_CASSERT(g == grad.host() + grad.size());
            return scale * loss;
        }
        
        friend void serialize(const loss_softmax_log_&, std::ostream& out) {
            dlib::serialize("loss_softmax_log_", out);
        }

        friend void deserialize(loss_softmax_log_&, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "loss_softmax_log_")
                throw dlib::serialization_error("Unexpected version found while deserializing dlib::loss_softmax_log_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_softmax_log_&) {
            out << "loss_softmax_log";
            return out;
        }

        friend void to_xml(const loss_softmax_log_&, std::ostream& out) {
            out << "<loss_softmax_log/>";
        }
    }; 

    template <typename SUBNET>
    using loss_softmax_log = dlib::add_loss_layer<loss_softmax_log_, SUBNET>;
}


