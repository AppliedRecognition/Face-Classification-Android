#pragma once

#include <dlib/dnn/input.h>
#include <algorithm>

namespace dlibx {

    /** \brief Input normalization options.
     *
     * These values are serialized as part of model format so don't
     * change existing values.
     */
    enum class input_normalization {
        none = 0,
        zero_center = 1,   // output range is [-1,1]
        minmax = 2,        // extend values to fill output range
        minmax_zero_center = 3
    };

    /** \brief Input layer accepting any image matching generic image interface.
     *
     * Template argument is whatever input_type you want trainer to accept.
     *
     * Compatible with image<matrix> and image<array2d>.
     * Serializes as image<matrix>.
     *
     * Supports optional normalization, but note that with normalization
     * this layer is no longer compatible with dlib::image.
     */
    template <typename image_type>
    class input_generic_image {
    public:
        using input_type = image_type;
        using pixel_type = typename dlib::image_traits<image_type>::pixel_type;
        static constexpr auto num_channels =
            dlib::pixel_traits<pixel_type>::num;
        static_assert(num_channels > 0, "pixel has no channels");

        input_generic_image(
            input_normalization norm = input_normalization::none)
            : norm(norm) {}

        template <typename T>
        input_generic_image(const input_generic_image<T>&) {
            static_assert(num_channels == input_generic_image<T>::num_channels,
                          "num_channels mismatch in copy constructor");
        }

        template <typename T>
        input_generic_image(const dlib::input<T>&) {
            static_assert(num_channels == input_generic_image<T>::num_channels,
                          "num_channels mismatch in copy constructor");
        }

        inline auto get_input_normalization() const {
            return norm;
        }

        bool image_contained_point(
            const dlib::tensor& data, const dlib::point& p) const {
            return get_rect(data).contains(p);
        }
        dlib::drectangle tensor_space_to_image_space(
            const dlib::tensor&, dlib::drectangle r) const {
            return r;
        }
        dlib::drectangle image_space_to_tensor_space(
            const dlib::tensor&, double, dlib::drectangle r) const {
            return r;
        }

        template <typename forward_iterator>
        void to_tensor(forward_iterator ibegin, forward_iterator iend,
                       dlib::resizable_tensor& data) const {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto nr = num_rows(*ibegin);
            const auto nc = num_columns(*ibegin);
            DLIB_CASSERT(nr > 0 && nc > 0,
                         "\t input_generic_image::to_tensor()"
                         << "\n\t Images given to to_tensor() must positive dimensions."
                         << "\n\t nr: " << nr
                         << "\n\t nc: " << nc
                );
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i) {
                DLIB_CASSERT(num_rows(*i) == nr && num_columns(*i) == nc,
                             "\t input_generic_image::to_tensor()"
                             << "\n\t All images given to to_tensor() must have the same dimensions."
                             << "\n\t nr: " << nr
                             << "\n\t nc: " << nc
                             << "\n\t i->nr(): " << num_rows(*i)
                             << "\n\t i->nc(): " << num_columns(*i)
                    );
            }

            using itype = std::decay_t<decltype(*ibegin)>;
            using ptype = typename dlib::image_traits<itype>::pixel_type;
            static_assert(num_channels == dlib::pixel_traits<ptype>::num,
                          "image passed to to_tensor() has incorrect number of channels");
            using bptype = typename dlib::pixel_traits<ptype>::basic_pixel_type;
            static constexpr auto is_byte =
                std::is_same<bptype, unsigned char>::value;

            // allocate memory in data tensor
            data.set_size(std::distance(ibegin,iend), num_channels, nr, nc);

            using in = input_normalization;
            const bool zc = int(norm) & int(in::zero_center);
            const bool mm = int(norm) & int(in::minmax);

            const auto stride = std::size_t(nr)*std::size_t(nc);
            auto dest = data.host();
            for (auto i = ibegin; i != iend; ++i) {
                auto src = static_cast<const char*>(image_data(*i));
                for (long r = 0; r < nr; src += width_step(*i), ++r) {
                    auto px = reinterpret_cast<const ptype*>(src);
                    for (long c = 0; c < nc; ++dest, ++px, ++c) {
                        const auto temp = dlib::pixel_to_vector<float>(*px);
                        auto p = dest;
                        for (long j = 0; j < num_channels; p += stride, ++j)
                            *p = is_byte ? temp(j)/256.0f : temp(j);
                    }
                }
                dest += stride*(num_channels-1);
                if (zc || mm) {
                    auto first = dest - stride*num_channels;
                    // technically we should have max = 255.0f / 256.0f,
                    // but max = 1 allows an input pixel value
                    // of 128 to map to exactly 0.0f in the tensor
                    // (unless minmax is being used)
                    float min = 0, max = 1;
                    if (mm) {
                        const auto mmel = std::minmax_element(first, dest);
                        min = *mmel.first;
                        max = *mmel.second;
                    }
                    if (min < max) {
                        float sub, scale;
                        if (zc) {
                            // y = (x - max/2 - min/2) * 2 / (max-min);
                            sub = (min+max) / 2.0f;
                            scale = 2.0f / (max-min);
                        }
                        else {
                            // y = (x - min) / (max - min)
                            sub = min;
                            scale = 1.0f / (max-min);
                        }
                        for ( ; first != dest; ++first)
                            *first = (*first - sub) * scale;
                    }
                }
            }
        }

        friend void
        serialize(const input_generic_image& obj, std::ostream& out) {
            if (obj.norm == input_normalization::none)
                dlib::serialize("input<matrix>", out);
            else {
                dlib::serialize("input_generic_image", out);
                dlib::serialize(num_channels, out);
                dlib::serialize(int(obj.norm), out);
            }
        }

        friend void deserialize(input_generic_image& obj, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version == "input_generic_image") {
                int nc, norm;
                dlib::deserialize(nc, in);
                dlib::deserialize(norm, in);
                if (nc != num_channels)
                    throw dlib::serialization_error("Incorrect number of channels found while deserializing dlibx::input_generic_image.");
                obj.norm = input_normalization(norm);
            }
            else if (version == "input<matrix>" ||
                     version == "input<array2d>")
                obj.norm = input_normalization::none;
            else
                throw dlib::serialization_error("Unexpected version found while deserializing dlibx::input_generic_image.");
        }

        friend void to_xml(const input_generic_image& obj, std::ostream& out) {
            if (obj.norm == input_normalization::none)
                out << "<input/>";
            else
                out << "<input normalization='" << int(obj.norm) << "'/>";
        }

        friend std::ostream&
        operator<<(std::ostream& out, const input_generic_image& obj) {
            if (obj.norm == input_normalization::none)
                out << "input_generic_image";
            else
                out << "input_generic_image\t (normalization="
                    << int(obj.norm) << ')';
            return out;
        }

    private:
        input_normalization norm;
    };
}
