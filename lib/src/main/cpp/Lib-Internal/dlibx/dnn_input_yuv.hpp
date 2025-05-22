#pragma once

#include <dlib/dnn/input.h>
#include <raw_image/types.hpp>
#include <stdext/arg.hpp>

namespace dlibx {
    
    /** \brief Input layer accepting YUV24 raw_image::plane and doing
     * brightness and contrast normalization.
     *
     * The Y channel will be normalized to mean zero and standard deviation one.
     * The U and V channels are will have values in the range [-1.0, 1.0).
     */
    class input_yuv_normalized {
    public:
        using input_type = raw_image::plane;
        static constexpr auto num_channels = 3;

        input_yuv_normalized() = default;

        bool image_contained_point(
            const dlib::tensor& data, const dlib::point& p) const {
            return get_rect(data).contains(p);
        }
        dlib::drectangle tensor_space_to_image_space(
            const dlib::tensor&, dlib::drectangle r) const { return r; }
        dlib::drectangle image_space_to_tensor_space(
            const dlib::tensor&, double, dlib::drectangle r) const { return r; }

        template <typename forward_iterator>
        void to_tensor(forward_iterator ibegin, forward_iterator iend,
                       dlib::resizable_tensor& data) const {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto first =
                stdx::pointer_to<const raw_image::plane>(*ibegin);
            DLIB_CASSERT(first);
            const auto nr = first->height;
            const auto nc = first->width;
            DLIB_CASSERT(nr > 0 && nc > 0,
                         "\t input_yuv_normalized::to_tensor()"
                         << "\n\t Images given to to_tensor() must have positive dimensions."
                         << "\n\t nr: " << nr
                         << "\n\t nc: " << nc
                );

            // allocate memory in data tensor
            data.set_size(std::distance(ibegin,iend), num_channels, nr, nc);

            const auto csize = std::size_t(nr)*std::size_t(nc);
            auto dest = data.host();
            for ( ; ibegin != iend; ++ibegin) {
                const auto imgp =
                    stdx::pointer_to<const raw_image::plane>(*ibegin);
                DLIB_CASSERT(imgp);
                auto& img = *imgp;
                DLIB_CASSERT(
                    same_channel_order(img.layout, raw_image::pixel::yuv),
                    "\t input_yuv_normalized::to_tensor()"
                    << "\n\t Images must have YUV pixel layout."
                    );
                DLIB_CASSERT(img.height == nr && img.width == nc,
                             "\t input_yuv_normalized::to_tensor()"
                             << "\n\t All images given to to_tensor() must have the same dimensions."
                             << "\n\t nr: " << nr
                             << "\n\t nc: " << nc
                             << "\n\t i->nr(): " << img.height
                             << "\n\t i->nc(): " << img.width
                    );
                uint64_t sum = 0, s2 = 0;
                unsigned char const* src = img.data;
                for (auto r = nr; r > 0; src += img.bytes_per_line, --r) {
                    auto yp = src;
                    for (auto c = nc; c > 0; yp += 3, --c) {
                        sum += *yp;
                        s2 += unsigned(*yp) * *yp;
                    }
                }
                const auto mean = float(double(sum) / double(csize));
                const auto m2 = unsigned(std::lround(mean*mean));
                s2 /= csize;
                const auto coeff =
                    s2 <= m2 ? 1.0f : 1.0f / float(std::sqrt(s2-m2));
                src = img.data;
                for (auto r = nr; r > 0; src += img.bytes_per_line, --r) {
                    auto sp = src;
                    for (auto c = nc; c > 0; sp += 3, ++dest, --c) {
                        // y
                        dest[0] = coeff * (float(sp[0]) - mean);
                        // u and v
                        dest[csize] = float(sp[1] - 128) / 128;
                        dest[2*csize] = float(sp[2] - 128) / 128;
                    }
                }
                dest += csize * (num_channels-1);
            }
        }

        friend void serialize(const input_yuv_normalized&, std::ostream& out) {
            dlib::serialize("input_yuv_normalized", out);
        }

        friend void deserialize(input_yuv_normalized&, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "input_yuv_normalized")
                throw dlib::serialization_error("Unexpected version found while deserializing dlibx::input_yuv_normalized.");
        }

        friend void to_xml(const input_yuv_normalized&, std::ostream& out) {
            out << "<input_yuv_normalized/>";
        }

        friend std::ostream&
        operator<<(std::ostream& out, const input_yuv_normalized&) {
            return out << "input_yuv_normalized";
        }
    };
}
