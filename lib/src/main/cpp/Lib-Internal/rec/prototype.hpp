#pragma once

#include "types.hpp"
#include "serialize.hpp"
#include <json/types.hpp>
#include <core/context.hpp>
#include <raw_image/types.hpp>
#include <stdext/arg.hpp>
#include <stdext/span.hpp>
#include <stdext/forward_iterator.hpp>
#include <optional>
#include <vector>
#include <array>


namespace rec {
    using multi_plane_arg = stdx::spanarg<const raw_image::plane>;

    /** \brief Compare prototypes and return score.
     */
    compare_result
    compare(stdx::arg<const prototype> a, stdx::arg<const prototype> b,
            variant var = variant::none);


    /** \brief Transcribe prototype to different version.
     *
     * If the supplied prototype already has the requested version then
     * a copy of it is returned instead.
     *
     * \throws std::runtime_error if the conversion is not supported
     */
    prototype_ptr
    transcribe(stdx::arg<const core::context_data> context,
               stdx::arg<const prototype> proto,
               version_type target_version);
    

    /** \brief Compute rotated bounding box to use for extraction.
     */
    rotated_box bounding_box(
        stdx::arg<core::context_data> context,
        const det::face_coordinates& coordinates,
        version_type version);

    /** \brief Extract prototype from image.
     */
    prototype_ptr extract(
        core::active_job job,
        const multi_plane_arg& image,
        const rotated_box& rbox,
        version_type version,
        const json::object& settings = {});

    /** \brief Extract prototype from image.
     */
    prototype_ptr extract(
        core::active_job job,
        const multi_plane_arg& image,
        const det::face_coordinates& coordinates,
        version_type version,
        const json::object& settings = {});


    /** \brief Abstract base for recognition prototype (a.k.a. template).
     */
    class prototype {

    public:
        /** \brief Force loading of model files needed for extraction.
         *
         * Calling this method is optional.
         * The necessary model files will be loaded by extract() if necessary.
         */
        static void load_model(
            stdx::arg<core::context> context, version_type version);

        /** \brief Extract prototype from image.
         *
         * This static method kept for backward compatibility.
         * Use the rec::extract() version in new code.
         */
        template <typename U>
        static inline prototype_ptr extract(
            U&& job,
            const multi_plane_arg& image,
            const det::face_coordinates& coordinates,
            version_type version,
            const json::object& settings = {}) {
            return rec::extract(
                std::forward<U>(job), image, coordinates, version, settings);
        }

        /** \brief Extract jittered set of prototypes from image.
         *
         * The first element in the returned vector will be
         * the canonical "central" prototype.
         * The additional elements are the jittered prototypes in
         * no particular order.
         *
         * The jitter_options object has the following structure: <pre>
         *   {
         *     "roll":  <int>,   # roll degrees clockwise and ccw
         *     "horz":  <int>,   # horz pixel shift left and right
         *     "vert":  <int> or <array of int>,    # vert pixel shift
         *     "scale": <int> or <array of int>,    # scale by exp(int/64)
         *     "contrast": <int> or <array of int>, # contrast delta
         *     "cbase": <int>,   # base contrast (default is 48)
         *   }
         * </pre>
         * All values are optional.
         */
        static std::vector<prototype_ptr> jitter(
            core::active_job job,
            const multi_plane_arg& image,
            const det::face_coordinates& coordinates,
            version_type version,
            const json::object& jitter_options);

        /** \brief Deserialize.
         *
         * Value may be one of the output of to_json(), to_binary(), or
         * to_binary() as a base64 encoded string.
         *
         * This method can deserialize either a single prototype or a
         * multiface/subject object containing a single prototype.
         */
        static prototype_ptr
        deserialize(stdx::arg<const core::context_data> context,
                    const void* src, std::size_t len);
        static prototype_ptr
        deserialize(stdx::arg<const core::context_data> context,
                    const json::value& val);

        /** \brief Set serialize format for prototypes of specific version.
         *
         * The format code is version specific but generally 0 means default.
         */
        static void set_serialize_format(
            stdx::arg<core::context_data> context,
            version_type version,
            int format);

        /** \brief Set default comparison variant for specified version.
         */
        static void set_comparison_variant(
            stdx::arg<core::context_data> context,
            version_type version,
            variant var);

        /** \brief Construct random (meaningless) prototype.
         *
         * These methods are meant for use in unit tests.
         *
         * The second version creates a new prototype that will compare to
         * an existing prototype with the specified score.
         */
        static prototype_ptr random(
            core::active_job job,
            version_type version);
        static prototype_ptr random(
            core::active_job job,
            stdx::arg<const prototype> base,
            float score, variant var = variant::none);


    public:
        /** \brief Version number.
         */
        const version_type version;

        /** \brief 128-bit uuid.
         *
         * This is a non-cryptographically secure hash.
         */
        const uuid_type uuid;


        virtual ~prototype() = default;


        /** \brief Get diagnostic image.
         *
         * Will return nullptr if the particular diagnostic is not supported,
         * not available, or otherwise cannot be generated.
         *
         * If the selected diagnostic requires context data and it has
         * not been provided, then nullptr is returned.
         *
         * While the raw_image::plane object returned will be owned by
         * the caller, the pixel data pointed to may be owned by this
         * prototype object.
         */
        virtual raw_image::plane_ptr
        diagnostic_image(diagnostic type,
                         core::context_data* context = nullptr) const = 0;
        inline raw_image::plane_ptr
        diagnostic_image(stdx::arg<core::context_data> context,
                         diagnostic type) const {
            return diagnostic_image(type, context.get());
        }


        /** \brief Get raw projection vector.
         *
         * The returned vector will have norm 1.0,
         * give or take rounding error.
         */
        friend std::vector<float> to_float_vector(stdx::arg<const prototype>);
        

        /** \brief Transcribe prototype to different version.
         */
        friend prototype_ptr
        transcribe(stdx::arg<const core::context_data> context,
                   stdx::arg<const prototype> proto,
                   version_type target_version);
            

        /** \brief Compare prototypes and return score.
         */
        friend compare_result
        compare(stdx::arg<const prototype> a, stdx::arg<const prototype> b,
                variant var);


        /** \brief Serialize to json object.
         */
        friend json::value to_json(stdx::arg<const prototype>);

        /** \brief Serialize to binary.
         *
         * Don't use this method directly.
         * Use to_binary() instead (defined in types.hpp).
         *
         * Default is uncompressed amf3.
         */
        friend stdx::binary to_binary_with_opts(
            stdx::arg<const prototype>,
            stdx::options_tuple<serialize_type,compression_type>);


    protected:
        /** \brief Serialize to binary.
         */
        virtual stdx::binary serialize() const = 0;

        /** \brief Iterator and size to construct vector for use
         * by pca distance method.
         */
        std::pair<stdx::forward_iterator<float>,unsigned>
        virtual vector_for_pca(unsigned i = 0) const = 0;

        /** \brief Transcribe to other version.
         * \throws exception for unsupported conversion
         */
        virtual prototype_ptr
        transcribe_to(const core::context_data& context,
                      version_type target_version) const = 0;

        /** \brief Compare to other.
         *
         * \pre other.version == version
         */
        virtual compare_result
        compare_to(const prototype& other, variant var) const = 0;
        
        /** \brief Deep copy possibly with altered uuid.
         */
        virtual prototype_ptr
        copy(const std::optional<uuid_type>& uuid = {}) const = 0;

        /** \brief Construct multiface object for prototypes of same version.
         */
        virtual std::unique_ptr<internal::multiface>
        construct_multiface(float cluster_threshold) const = 0;
        friend class multiface;

        explicit prototype(version_type version, const uuid_type& uuid)
            : version(version), uuid(uuid) {}

        
    private:
        static prototype_ptr
        deserialize_bin(const core::context_data& context,
                        const void* src, std::size_t len,
                        const std::optional<uuid_type>& uuid = {});

        prototype(prototype&&) = delete;
        prototype(const prototype&) = delete;
        prototype& operator=(prototype&&) = delete;
        prototype& operator=(const prototype&) = delete;
    };
}
