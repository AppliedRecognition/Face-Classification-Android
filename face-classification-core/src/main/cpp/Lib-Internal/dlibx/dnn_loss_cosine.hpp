#pragma once

#include <dlib/dnn/utilities.h>  // for log1pexp() needed by loss.h
#include <dlib/dnn/loss.h>

namespace dlibx {
    /** \brief Cosine distance loss function.
     */
    class loss_cosine_ {
    public:
        using training_label_type = unsigned long;
        using output_label_type = dlib::matrix<float,0,1>;

        loss_cosine_() = default;

        loss_cosine_(float margin) : margin(margin) {
            DLIB_CASSERT(margin >= 0);
        }

        float get_margin() const {
            return margin;
        }

        template <typename SUB_TYPE, typename label_iterator>
        void to_label(const dlib::tensor& input, const SUB_TYPE& sub,
                      label_iterator iter) const {
            const dlib::tensor& output = sub.get_output();
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input.num_samples() != 0);
            DLIB_CASSERT(input.num_samples() % sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input.num_samples() == output.num_samples());
            DLIB_CASSERT(output.nr() == 1 && output.nc() == 1);

            const float* p = output.host();
            for (long i = 0; i < output.num_samples(); ++i) {
                *iter = dlib::mat(p,output.k(),1);
                ++iter;
                p += output.k();
            }
        }

        template <typename const_label_iterator, typename SUBNET>
        double compute_loss_value_and_gradient(const dlib::tensor& input,
                                               const_label_iterator truth,
                                               SUBNET& sub) const {
            // note: input is input to entire neural net and not used here
            const dlib::tensor& embedding = sub.get_output();
            dlib::tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input.num_samples() > 0);
            DLIB_CASSERT(input.num_samples() % sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input.num_samples() == embedding.num_samples());

            const std::vector<training_label_type> labels(
                truth, truth + embedding.num_samples());

            last_gradient = &grad;
            return compute_loss_value_and_gradient(
                embedding, labels.data(), grad);
        }

        friend void serialize(const loss_cosine_& item, std::ostream& out) {
            dlib::serialize("loss_cosine_1", out);
            dlib::serialize(item.margin, out);
        }

        friend void deserialize(loss_cosine_& item, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version == "loss_cosine_1")
                dlib::deserialize(item.margin, in);
            else
                throw dlib::serialization_error("Unexpected version found while deserializing dlibx::loss_cosine_.  Instead found " + version);
        }

        friend std::ostream&
        operator<<(std::ostream& out, const loss_cosine_& item) {
            out << "loss_cosine (margin=" << item.margin << ")";
            return out;
        }

        friend void to_xml(const loss_cosine_& item, std::ostream& out) {
            out << "<loss_cosine margin='" << item.margin << "'/>";
        }


        mutable dlib::tensor const* last_gradient = nullptr; ///< for debugging
        
    private:
        float margin = 0.05f;

        double compute_loss_value_and_gradient(
            const dlib::tensor& embedding,
            training_label_type const* labels,
            dlib::tensor& grad) const;
    };

    template <typename SUBNET>
    using loss_cosine = dlib::add_loss_layer<loss_cosine_, SUBNET>;
}

