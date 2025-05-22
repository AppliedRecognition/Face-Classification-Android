#pragma once

#include <dlib/dnn/input.h>

namespace dlibx {
    
    /** \brief Input layer accepting an array or vector of grayscale images.
     *
     * Compatible with and serializes as image<array<matrix>>.
     *
     * Each set of frames must be a container that supports std::begin()
     * and each image must have the generic image interface.
     */
    template <std::size_t K>
    class input_frames {
    public:
        using input_type = std::array<dlib::matrix<unsigned char>,K>;
        static_assert(K > 0, "input has no channels");

        input_frames() = default;
        input_frames(const input_frames&) = default;

        template <typename T, long NR, long NC, typename MM, typename L>
        input_frames(const dlib::input<std::array<dlib::matrix<T,NR,NC,MM,L>,K> >&) {
        }

        bool image_contained_point(const dlib::tensor& data, const dlib::point& p) const { return get_rect(data).contains(p); }
        dlib::drectangle tensor_space_to_image_space(const dlib::tensor&, dlib::drectangle r) const { return r; }
        dlib::drectangle image_space_to_tensor_space(const dlib::tensor&, double, dlib::drectangle r) const { return r; }

        template <typename C>
        static constexpr auto size(const C& c) {
            const auto n = std::distance(std::begin(c), std::end(c));
            return std::size_t(n >= 0 ? n : 0);
        }

        template <typename forward_iterator>
        void to_tensor(forward_iterator ibegin, forward_iterator iend,
                       dlib::resizable_tensor& data) const {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            DLIB_CASSERT(std::begin(*ibegin) != std::end(*ibegin),
                         "When using std::array<matrix> inputs you can't give 0 sized arrays.");
            const auto nr = num_rows(*std::begin(*ibegin));
            const auto nc = num_columns(*std::begin(*ibegin));
            DLIB_CASSERT(nr > 0 && nc > 0,
                         "\t input_frames::to_tensor()"
                         << "\n\t Images given to to_tensor() must have positive dimensions."
                         << "\n\t nr: " << nr
                         << "\n\t nc: " << nc
                );
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i) {
                DLIB_CASSERT(size(*i) == K,
                             "\t input_frames::to_tensor()"
                             << "\n\t Incorrect number channels."
                             << "\n\t Expected: " << K
                             << "\n\t Found: " << std::distance(std::begin(*i), std::end(*i))
                    );
                for (auto j = std::begin(*i); j != std::end(*i); ++j) {
                    DLIB_CASSERT(num_rows(*j) == nr && num_columns(*j) == nc,
                                 "\t input_frames::to_tensor()"
                                 << "\n\t All images given to to_tensor() must have the same dimensions."
                                 << "\n\t nr: " << nr
                                 << "\n\t nc: " << nc
                                 << "\n\t j->nr(): " << num_rows(*j)
                                 << "\n\t j->nc(): " << num_columns(*j)
                        );
                }
            }

            using itype = std::decay_t<decltype(*std::begin(*ibegin))>;
            using ptype = typename dlib::image_traits<itype>::pixel_type;
            static_assert(1 == dlib::pixel_traits<ptype>::num,
                          "frame passed to to_tensor() must be grayscale");
            using bptype = typename dlib::pixel_traits<ptype>::basic_pixel_type;
            static constexpr auto is_byte =
                std::is_same<bptype, unsigned char>::value;
            
            // allocate memory in data tensor
            data.set_size(std::distance(ibegin,iend), K, nr, nc);

            auto dest = data.host();
            for (auto i = ibegin; i != iend; ++i) {
                for (auto j = std::begin(*i); j != std::end(*i); ++j) {
                    for (long r = 0; r < nr; ++r) {
                        for (long c = 0; c < nc; ++c) {
                            if (is_byte)
                                *dest++ = (*j)(r,c)/256.0f;
                            else
                                *dest++ = (*j)(r,c);
                        }
                    }
                }
            }
        }

        friend void serialize(const input_frames&, std::ostream& out) {
            dlib::serialize("input<array<matrix>>", out);
        }

        friend void deserialize(input_frames&, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "input<array<matrix>>")
                throw dlib::serialization_error("Unexpected version found while deserializing dlib::input<array<matrix>>.");
        }

        friend std::ostream& operator<<(std::ostream& out, const input_frames&) {
            out << "input<array<matrix>>";
            return out;
        }

        friend void to_xml(const input_frames&, std::ostream& out) {
            out << "<input/>";
        }
    };
}
