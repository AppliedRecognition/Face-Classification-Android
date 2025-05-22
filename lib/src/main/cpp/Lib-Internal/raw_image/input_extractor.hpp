#pragma once

#include "scaled_chip.hpp"
#include <string_view>
#include <vector>

namespace raw_image {

    /** \brief Abstract base class for neural net input extraction.
     */
    class input_extractor {
    public:
        const std::string name;
        const unsigned width;
        const unsigned height;
        const pixel_layout layout;

        /** \brief Destructor.
         */
        virtual ~input_extractor() = default;

        /** \brief Compute chip_details object from landmarks.
         */
        inline scaled_chip
        operator()(const std::vector<point2f>& pts) const {
            return chip_from_pts(pts);
        }

        /** \brief Extract image chip from chip_details object.
         */
        raw_image::plane_ptr
        operator()(const raw_image::multi_plane_arg& image,
                   const scaled_chip& chip) const {
            return extract_from_chip(image,chip);
        }

        /** \brief Extract image chip from landmarks.
         */
        raw_image::plane_ptr
        operator()(const raw_image::multi_plane_arg& image,
                   const std::vector<point2f>& pts) const {
            return extract_from_pts(image,pts);
        }

        /** \brief Find or construct an extractor with specified layout.
         *
         * This method will attempt to produce an extractor to crop the
         * same region of the image but return a result with a different
         * pixel layout.
         */
        const input_extractor* new_layout(raw_image::pixel_layout) const;

        /** \brief Find previously registered extractor or construct
         * from factory.
         *
         * See the documentation in the files input_extractor_*.hpp
         * for the string structure required for specific extractors.
         *
         * \returns nullptr if extractor not found
         */
        static const input_extractor* find(std::string_view name);

        /** \brief Register factory method for specified prefix.
         */
        using unique_ptr = std::unique_ptr<const input_extractor>;
        using factory_method = unique_ptr(*)(const std::string_view&);
        static void
        register_factory(std::string prefix, factory_method factory);


    protected:
        virtual scaled_chip
        chip_from_pts(const std::vector<point2f>& pts) const = 0;

        /// The default version calls extract_image_chip().
        virtual raw_image::plane_ptr
        extract_from_chip(const raw_image::multi_plane_arg& image,
                          const scaled_chip& chip) const {
            return extract_image_chip(image, chip, layout);
        }

        /// The default version does extract_from_chip(chip_from_pts(...)).
        virtual raw_image::plane_ptr
        extract_from_pts(const raw_image::multi_plane_arg& image,
                         const std::vector<point2f>& pts) const {
            return extract_from_chip(image, chip_from_pts(pts));
        }

        input_extractor(std::string name, unsigned width, unsigned height,
                        raw_image::pixel_layout layout)
            : name(move(name)), width(width), height(height),
              layout(layout) {
        }
        input_extractor(const input_extractor&) = delete;
    };
}
