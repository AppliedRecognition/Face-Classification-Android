#pragma once

#include "types.hpp"
#include <core/context.hpp>
#include <raw_image/core.hpp>

namespace det {

    /** \brief Rotate option.
     *
     * If rotate > 0, rotate image by multiple of 90 degrees.
     * If rotate & 4, mirror image before rotation.
     */
    enum class rotate : unsigned {};

    /** \brief Grayscale option.
     *
     * Load image in grayscale or convert to grayscale if color is not
     * needed for image detection.
     * Currently, the v3 (hog) detector does not require color.
     * Also, landmark detection also does not require color.
     *
     * This option may be used to speed up image loading or reduce memory usage.
     */
    struct gray_tag;
    using gray_option = stdx::option_bool<gray_tag>;
    const gray_option gray{true};

    /** \brief Color option.
     *
     * Preserve the color version of the input image (if it is color).
     *
     * If both the gray and color options are specified
     * then the returned image may have both.  
     * Note that they may be rotated differently.
     *
     * If neither option is specified, the returned image will contain
     * one or the other as determined by the input image and detection
     * settings.
     */
    struct color_tag;
    using color_option = stdx::option_bool<color_tag>;
    const color_option color{true};


    namespace internal {
        /* These methods are the internal implemention which
         * should not be called directly.
         * Use the methods outside the "internal" namespace instead (below).
         */
        image_type take_image(
            core::active_job context,
            const detection_settings& settings,
            std::unique_ptr<raw_image::plane> raw_image,
            const stdx::options_tuple<gray_option,color_option>& opts);
        image_type share_image(
            core::active_job context,
            const detection_settings& settings,
            std::shared_ptr<const raw_image::plane> raw_image,
            const stdx::options_tuple<gray_option,color_option>& opts);
        image_type use_pixels(
            core::active_job context,
            const detection_settings& settings,
            const raw_image::plane& raw_image,
            const stdx::options_tuple<gray_option,color_option>& opts);
        image_type share_pixels(
            core::active_job context,
            const detection_settings& settings,
            const raw_image::multi_plane_arg& raw_image,
            const stdx::options_tuple<gray_option,color_option>& opts);
        image_type copy_image(
            core::active_job context,
            const detection_settings& settings,
            const raw_image::multi_plane_arg& raw_image,
            const stdx::options_tuple<gray_option,color_option>& opts);
        const raw_image::plane&
        get_raw_from_image(
            stdx::arg<const image_struct> image,
            const stdx::options_tuple<gray_option,color_option>& opts);
    }


    /** \name Image setup.
     *
     * Comparison of image construction methods:
     *
     *  (old names)  load          move   use    take    use
     *               image         image  image  pixels  pixels
     *
     *  (new names)         copy_  take_  share  use_    share_
     *                      image  image  image  pixels  pixels
     *               -----  -----  -----  -----  ------  ------
     * constant time  no     no     yes    yes    yes     yes
     * may modify     yes    yes    yes    no     yes     no
     * owns pixels    yes    yes    yes?   yes?   no      no
     * multi plane    n/a    yes    no     no     no      yes
     *
     * Constant time means a shallow (metadata only) copy is made, and the pixel
     * data must not be modified or memory released until detection is complete.
     *
     * May modify means the pixel data may be modified in place by the detector
     * in any way, including changes in pixel layout or dimensions.  No threads
     * outside the detector should read the pixel buffer.
     *
     * Owns pixels means the returned image object has ownership of the pixel
     * buffer.  (?) ownership is only assured if the supplied unique_ptr
     * or shared_ptr owns the pixel buffer.  Obviously with a shared_ptr,
     * ownership is shared.
     *
     * Multi plane means a multi-plane image may be provided as input.
     * All other methods accept single plane images.
     *
     * Note that load_image() was removed so that det no longer depends
     * on raw_image_io.  The same functionality can be achieved with
     *   take_image(context, settings, raw_image::load(filename, ...)).
     */
    //@{
    /** \brief Move raw image into image.
     *
     * This method takes ownership of the image and may modify it in any way.
     *
     * The image rotate and scale values are respected / used by this method.
     */
    template <typename... Opts>
    inline image_type take_image(
        core::active_job context,
        const detection_settings& settings,
        std::unique_ptr<raw_image::plane> raw_image,
        Opts&&... opts) {
        return internal::take_image(
            std::move(context), settings, std::move(raw_image),
            { std::forward<Opts>(opts)... } );
    }
    template <typename... Opts>
    [[deprecated("Use take_image() instead.")]]
    inline image_type move_image(
        core::active_job context,
        const detection_settings& settings,
        std::unique_ptr<raw_image::plane> raw_image,
        Opts&&... opts) {
        return internal::take_image(
            std::move(context), settings, std::move(raw_image),
            { std::forward<Opts>(opts)... } );
    }
    
    /** \brief Use raw image in place.
     *
     * The pixels will not be modified by this method or during detection.
     *
     * The image rotate and scale values are respected / used by this method.
     */
    template <typename... Opts>
    inline image_type share_image(
        core::active_job context,
        const detection_settings& settings,
        std::shared_ptr<const raw_image::plane> raw_image,
        Opts&&... opts) {
        return internal::share_image(
            std::move(context), settings, std::move(raw_image),
            { std::forward<Opts>(opts)... } );
    }
    template <typename... Opts>
    [[deprecated("Use share_image() instead.")]]
    inline image_type use_image(
        core::active_job context,
        const detection_settings& settings,
        std::shared_ptr<const raw_image::plane> raw_image,
        Opts&&... opts) {
        return internal::share_image(
            std::move(context), settings, std::move(raw_image),
            { std::forward<Opts>(opts)... } );
    }

    /** \brief Use raw image pixels in place.
     *
     * The pixels may be modified by this method.
     * The pixels must not be modified or freed while detection is in progress.
     *
     * The image rotate and scale values are respected / used by this method.
     */
    template <typename... Opts>
    inline image_type use_pixels(
        core::active_job context,
        const detection_settings& settings,
        const raw_image::plane& raw_image,
        Opts&&... opts) {
        return internal::use_pixels(
            std::move(context), settings, raw_image,
            { std::forward<Opts>(opts)... } );
    }
    template <typename... Opts>
    [[deprecated("Use use_pixels() instead.")]]
    inline image_type take_pixels(
        core::active_job context,
        const detection_settings& settings,
        const raw_image::plane& raw_image,
        Opts&&... opts) {
        return internal::use_pixels(
            std::move(context), settings, raw_image,
            { std::forward<Opts>(opts)... } );
    }

    /** \brief Use raw image pixels in place.
     *
     * The pixels will not be modified by this method or during detection.
     * The pixels must not be modified or freed while detection is in progress.
     *
     * The image rotate and scale values are respected / used by this method.
     */
    template <typename... Opts>
    inline image_type share_pixels(
        core::active_job context,
        const detection_settings& settings,
        const raw_image::multi_plane_arg& raw_image,
        Opts&&... opts) {
        return internal::share_pixels(
            std::move(context), settings, raw_image,
            { std::forward<Opts>(opts)... } );
    }

    /** \brief Copy raw image into image.
     *
     * The image rotate and scale values are respected / used by this method.
     */
    template <typename... Opts>
    inline image_type copy_image(
        core::active_job context,
        const detection_settings& settings, 
        const raw_image::multi_plane_arg& raw_image,
        Opts&&... opts) {
        return internal::copy_image(
            std::move(context), settings, raw_image,
            { std::forward<Opts>(opts)... } );
    }
    
    /** \brief Suggest power of two image scaling for detection.
     *
     * If the return value is non-zero, then face detection can be completed
     * using a scaled down version of the image.  
     * For return value N, the image may be scaled to width/2^N, height/2^N.
     * Note that all coordinates associated with detected faces will need
     * to be scaled back up (by 2^N) after detection if coordinates relative
     * to the original image are required.
     * Also, using a scaled down image may adversely affect recognition
     * accuracy.
     */
    unsigned suggested_scaling(
        const detection_settings& settings, 
        const raw_image::image_size& original_size);

    /** \brief Get original image dimensions.
     *
     * If the image was rotated when loaded, then this method returns the
     * dimensions after rotation.
     */
    raw_image::image_size
    get_image_dimensions(stdx::arg<const image_struct> image);

    /** \brief Get raw image from image object.
     *
     * The returned image may be scaled or rotated.
     * The scale and rotate members of the returned object will be
     * set appropriately.  
     *
     * Use get_image_dimensions() to determine the original dimensions
     * of the image.
     *
     * The returned image may have any color_space. 
     * Provide gray or color option to indicate a preference.
     * With neither option, the preference is color.
     *
     * The user must not modify the returned image pixels.
     * The image pixels will be freed when the image object is destructed.
     */
    template <typename... Opts>
    inline const raw_image::plane&
    get_raw_from_image(stdx::arg<const image_struct> image, Opts&&... opts) {
        return internal::get_raw_from_image(
            image, { std::forward<Opts>(opts)... } );
    }
    //@}
}
