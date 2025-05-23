#pragma once

#include <dlib/geometry/vector.h>
#include <raw_image/types.hpp>
#include <raw_image/point2.hpp>
#include <raw_image/point_rounding.hpp>
#include <json/types.hpp>
#include <stdext/forward_iterator.hpp>
#include <stdext/arg.hpp>
#include <stdext/span.hpp>
#include <istream>


namespace dlib {
    class tensor;
    class resizable_tensor;
}

namespace raw_image {
    class input_extractor;
}

namespace dlibx {

    using fpoint = dlib::vector<float,2>;
    using point2f = raw_image::point2f;

    namespace net {

        class layer;
        using layer_ptr = std::unique_ptr<layer>;
        using layer_ptr_vector = std::vector<layer_ptr>;


        /** \brief Neural net implemented as a vector of layers along with
         * related metadata.
         *
         * This object takes ownership of a std::vector<layer_ptr>, 
         * does map_layers() on it, and provides methods to run
         * images forward through the net.
         *
         * Read-only access is provided to the layers.
         * Note that iterators dereference to const layer&, not a pointer.
         * To make changes to the layers, first do release_layers(),
         * make your changes, and then do set_layers().
         */
        class vector {
        public:
            json::object meta;
            std::vector<std::string> labels;
            raw_image::input_extractor const* input_extractor = nullptr;

            using iterator = stdx::forward_iterator<const layer&>;
            using const_iterator = stdx::forward_iterator<const layer&>;


            vector();
            ~vector();
            vector(vector&&);
            vector& operator=(vector&&);
            vector(const vector& other);
            vector& operator=(const vector& other);


            /** \brief Construct from vector of layers.
             */
            vector(layer_ptr_vector&& layers);


            /** \brief Load (deserialize) from stream.
             */
            vector(std::istream& in);
            vector(std::istream&& in) : vector(in) {}


            /** \brief Replace layers.
             */
            void set_layers(layer_ptr_vector&& new_layers);

            /** \brief Extract layers.
             */
            layer_ptr_vector release_layers();
            operator layer_ptr_vector() &&;


            /** \brief Number of layers and const access.
             */
            inline bool empty() const { return m_layers.empty(); }
            inline auto size() const { return m_layers.size(); }
            inline auto const& layers() const { return m_layers; }


            /** \brief Read-only (const) access to input and output layers.
             */
            inline auto& front() const { return *m_layers.front(); }
            inline auto& back() const { return *m_layers.back(); }


            /** \brief Read-only (const) iteration through layers.
             */
            inline const_iterator begin() const {
                return {
                    m_layers.begin(),
                    [](const auto& ptr) -> const layer& { return *ptr; }
                };
            }
            inline const_iterator end() const {
                return {
                    m_layers.end(),
                    [](const auto& ptr) -> const layer& { return *ptr; }
                };
            }


            /** \brief Complete description of model.
             *
             * Copy "meta", "input", "labels" and "net" description
             * into single json object.
             */
            json::object description() const;


            /** \brief Generate a concise description of neural net structure.
             *
             * Same as back().concise().
             */
            std::string concise() const;


            /** \brief Output layer type and size of returned vector.
             *
             * If "sig" with output of size 1
             * or "softmax" with output of size 2,
             * then this is a binary-classification net.
             *
             * If "softmax" and output size is at least 3,
             * then this is a multi-classification net.
             *
             * If "fc", then the net is probably a regression.
             */
            std::pair<std::string, unsigned long> output_type_and_size() const;


            /** \brief Input to extract() methods.
             *
             * The extract() method supports multi-plane images.
             * For example, first plane is y8_nv21 and second plane is vu16.
             * This is considered a single frame.
             */
            using multi_plane_arg = stdx::spanarg<const raw_image::plane>;

            /** \brief Extract image needed by neural network.
             *
             * \throws exception if input_extractor null or other error
             */
            raw_image::plane_ptr
            extract(const multi_plane_arg& image,
                    const std::vector<point2f>& pts) const;
            template <typename ITER>
            inline auto extract(const multi_plane_arg& image,
                                ITER pts_first, ITER pts_last) const {
                std::vector<point2f> pts;
                const auto n = pts_last - pts_first;
                if (n > 0) {
                    pts.reserve(std::size_t(n));
                    for ( ; pts_first != pts_last; ++pts_first)
                        pts.push_back(raw_image::round_from(*pts_first));
                }
                return extract(image, pts);
            }


            /** \brief Input to operator() methods.
             *
             * The apply model methods operator() support multi-frame inputs.
             * Note that even though the type is the same, these are not
             * multi-plane images.
             * Each raw_image::plane object must fully describe a singe frame.
             */
            using multi_frame_arg = stdx::spanarg<const raw_image::plane>;
            using multi_frame_span = stdx::span<const raw_image::plane>;

            /** \brief Apply single image to neural network.
             *
             * Note that even though this method accepts a span<plane>, 
             * multi-plane images such as Y8 + VU16 are not supported.
             * This interface will accept multi-frame images for cases
             * where the neural net is to process a sequence of video frames.
             *
             * If diagnostic is non-null, then per layer diagnostic
             * information will be provided.
             */
            [[deprecated("Use operator() with vector as dest arg instead.")]]
            auto operator()(const multi_frame_arg& img) {
                const auto p = apply(img,nullptr);
                return std::vector<float> { p.first, p.first + p.second };
            }
            inline void
            operator()(const multi_frame_arg& img, std::vector<float>& dest,
                       json::array* diagnostic = nullptr) {
                const auto p = apply(img,diagnostic);
                dest.assign(p.first, p.first + p.second);
            }
            template <std::size_t N>
            inline void
            operator()(const multi_frame_arg& img, std::array<float,N>& dest,
                       json::array* diagnostic = nullptr) {
                const auto p = apply(img,diagnostic);
                if (p.second != N)
                    throw std::logic_error(
                        "neural net did not return vector of correct size");
                std::copy_n(p.first, p.second, dest.data());
            }
            inline void
            operator()(const multi_frame_arg& img, float& dest,
                       json::array* diagnostic = nullptr) {
                const auto p = apply(img,diagnostic);
                if (p.second != 1)
                    throw std::logic_error(
                        "neural net did not return vector of correct size (single element expected)");
                dest = *p.first;
            }

            /** \brief Single sample input to multiple outputs.
             *
             * For models that produce multiple outputs (e.g. RetinaFace).
             * The caller is expected to know how may outputs will be
             * produced.  It is not an error if dest is the wrong size.
             * Either fewer outputs will be returned or some outputs
             * will not be computed.
             *
             * \returns number of outputs actually stored
             */
            std::size_t
            operator()(const multi_frame_span& img,
                       stdx::span<dlib::resizable_tensor> dest);

            /** \brief Use specified tensor as input instead of image.
             */
            std::size_t
            operator()(const dlib::tensor& input,
                       stdx::span<dlib::resizable_tensor> dest);
            

            /** \brief Apply multiple images to neural network.
             *
             * Each image may be one or more frames.
             * Multi-plane images (e.g. Y8 + VU16) are not supported.
             *
             * The result is multiple vectors, one per input image.
             *
             * This method performs the same calculation as running
             * the images through one-by-one, but is more efficient.
             */
            inline void
            operator()(stdx::forward_iterator<multi_frame_span> first,
                       stdx::forward_iterator<multi_frame_span> last,
                       std::vector<std::vector<float> >& dest) {
                auto n = distance(first, last);
                if (n > 0) {
                    auto p = apply(first, last);
                    dest.resize(std::size_t(n));
                    for (auto& vec : dest) {
                        auto end = p.first + p.second;
                        vec.assign(p.first, end);
                        p.first = end;
                    }
                }
                else
                    dest.clear();
            }
            template <std::size_t N>
            inline void
            operator()(stdx::forward_iterator<multi_frame_span> first,
                       stdx::forward_iterator<multi_frame_span> last,
                       std::vector<std::array<float,N> >& dest) {
                auto n = distance(first, last);
                if (n > 0) {
                    auto p = apply(first, last);
                    if (p.second != N)
                        throw std::logic_error(
                            "neural net did not return vector of correct size");
                    const auto pa =
                        reinterpret_cast<std::array<float,N> const*>(p.first);
                    dest.assign(pa, pa + n);
                }
                else
                    dest.clear();
            }
            void operator()(stdx::forward_iterator<multi_frame_span> first,
                            stdx::forward_iterator<multi_frame_span> last,
                            dlib::resizable_tensor& dest);


            friend inline void
            serialize(const vector& item, std::ostream& out) {
                item.serialize(out);
            }
            friend inline void
            deserialize(vector& item, std::istream& in) {
                item.deserialize(in);
            }

        private:
            layer_ptr_vector m_layers;

            std::pair<float const*, std::size_t>
            apply(const multi_frame_span&,json::array*);
            std::pair<float const*, std::size_t>
            apply(stdx::forward_iterator<multi_frame_span>&,
                  const stdx::forward_iterator<multi_frame_span>&);

            void serialize(std::ostream&) const;
            void deserialize(std::istream&);
        };
    }
}
