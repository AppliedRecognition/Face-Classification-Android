#pragma once

#include <raw_image/types.hpp>

namespace render {

    /** \brief Processing settings.
     */
    struct render_settings {
        /** \brief Lighting matrix to use for lighting compensation.
         *
         * Currently available matrices:<ul>
         * <li>3: generated from multi-pi (15 eigenvectors)</li>
         * </ul>
         *
         * Matrix 3 requires dlib landmarks.
         */
        unsigned lighting_matrix = 3;

        /** \brief Lighting compensation strength.
         *
         * A value of 0 results in only brightness and contrast correction.
         *
         * Values of 1 or greater select that number of eigenvectors
         * from the lighting matrix.
         * The higher values will provide greater compensation but may also
         * remove face features that are essential to recognition.
         */
        unsigned lighting_compensation = 0;

        /** \brief Pose compensation method.
         *
         * Currently available options:<ul>
         * <li>0: free form pose compensation</li>
         * <li>1: pose matrix mean face</li>
         * <li>2-7: multi-pie expression mean</li>
         * </ul>
         *
         * Options beyond 0 only apply with dlib landmarks (lighting_matrix 3).
         */
        unsigned pose_variant = 0;

        /** \brief Pose compensation strength.
         *
         * A value of 0 results in no face warping.
         *
         * Values of 1 or more (up to 10) select that number of eigenvectors
         * from the pose matrix.
         * Higher values provide a greater amount of face warping to make
         * arbitrary poses appear to be frontal.
         * Some tests indicate that a value of 4 is optimal for recognition.
         *
         * Pose compensation requires lighting_compensation > 0.
         */
        unsigned pose_compensation = 0;
    };

    /** \brief Output settings.
     */
    struct output_settings {
        /** \brief Output width.
         */
        unsigned width;

        /** \brief Output height.
         */
        unsigned height;

        /** \brief Distance between eyes as fraction of output width.
         */
        float eye_width;

        /** \brief Distance between top of image and the eyes as fraction of
         * output height.
         */
        float eye_vertical;

        /** \brief Output color space.
         */
        raw_image::pixel_layout color_space;
    };
}


