#pragma once

#include "types.hpp"
#include <core/context.hpp>
#include <stdext/arg.hpp>
#include <stdext/span.hpp>

namespace dlibx {
    namespace net {
        class vector;
    }
}
namespace det {

    /** \brief Classifier name, path and model.
     *
     * A classifier is a classification made by a neural net classifier.
     *
     * The path is for diagnostic purposes and is typically the result of
     * the <code>path.generic_string()</code> if a filesystem path was involved.
     * The path may be empty if the loader did not provide it.
     */
    struct classifier_model_type {
        std::string_view name;
        std::string path;
        const dlibx::net::vector& model;
    };

    /** \brief Load classifier model file.
     *
     * Classifiers are organized internally (per context) by name.
     * Only one classifier per name will be loaded and subsequent load
     * requests will return the previously loaded model.
     *
     * The classifier may be loaded from an open istream or deserialized
     * from a provided binary.
     * If neither of these are provided, then the internal model loader
     * method is used to find and load the classifier.
     */
    classifier_model_type const*
    load_classifier(stdx::arg<core::context> context,
                    std::string_view classifier_name);
    classifier_model_type const*
    load_classifier(stdx::arg<core::context> context,
                    std::string_view classifier_name,
                    stdx::arg<std::istream> from_stream,
                    std::string path = {});
    classifier_model_type const*
    load_classifier(stdx::arg<core::context> context,
                    std::string_view classifier_name,
                    stdx::binary from_binary,
                    std::string path = {});

    /** \brief Apply classifier to face.
     */
    std::vector<float>
    apply_classifier(stdx::arg<core::context> context,
                     classifier_model_type const* attr,
                     const stdx::spanarg<const raw_image::plane>& image,
                     const detected_coordinates& face);

    /** \brief Output constructor to apply classifiers.
     */
    class apply_classifiers {
        struct internal_config {
            std::vector<raw_image::plane> image;
            std::vector<const classifier_model_type*> detection_classifiers;
            std::vector<std::pair<const classifier_model_type*,float> > landmark_classifiers;
        };
        std::unique_ptr<internal_config> config;

        struct internal_state;
        std::unique_ptr<internal_state> state;

    public:
        ~apply_classifiers();
        apply_classifiers(apply_classifiers&&);
        apply_classifiers& operator=(apply_classifiers&&);
        //apply_classifiers(const apply_classifiers&);
        //apply_classifiers& operator=(const apply_classifiers&);

        apply_classifiers(
            const stdx::spanarg<const raw_image::plane>& image,
            std::vector<const classifier_model_type*> detection_classifiers,
            std::vector<std::pair<const classifier_model_type*,float> > landmark_classifiers);

        apply_classifiers(const apply_classifiers& other,
                          const face_coordinates& fc,
                          core::job_context& jc);

        face_coordinates_with_classifiers
        operator()(face_coordinates& fc, core::job_context& jc);

        stdx::span<const raw_image::plane> image() const;
    };
}
