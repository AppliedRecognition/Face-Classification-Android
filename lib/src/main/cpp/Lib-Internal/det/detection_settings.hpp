#pragma once

namespace json {
    class object;
    class value;
}
namespace det {

    /** \brief Landmark detection options.
     */
    enum class landmark_options : unsigned {
        none = 0, dlib5 = 2, dlib68 = 4, mesh68 = 8, mesh478 = 16
    };
    using lm = landmark_options;
    constexpr lm operator+(lm a, lm b) {
        return lm(unsigned(a) | unsigned(b));
    }
    constexpr lm operator|(lm a, lm b) {
        return lm(unsigned(a) | unsigned(b));
    }
    constexpr bool operator&(lm a, lm b) {
        return unsigned(a) & unsigned(b);
    }


    /** \brief Landmark detection settings.
     */
    struct landmark_settings {
        /** \brief Bitmask of landmarks to detect.
         */
        landmark_options landmarks = lm::dlib68;

        /** \brief Contrast correction setting.
         *
         * Only applies to dlib5 and dlib68 landmark detection.
         * Supported values are:<ul>
         *   <li>0 : no correction</li>
         *   <li>1 : correct to default value</li>
         * </ul>
         */
        unsigned contrast_correction = 0;


        constexpr landmark_settings() = default;
        constexpr landmark_settings(landmark_options landmarks)
            : landmarks(landmarks) {}
    };


    /** \brief Face detection settings (including landmark detection).
     */
    struct detection_settings {

        /** \brief Face detector version.
         *
         * Available versions are:<ul>
         * <li>3: dlib fhog detector</li>
         * <li>4: dlib resnet/cnn/dnn detector</li>
         * <li>5: face-api.js tiny face detector</li>
         * </ul>
         */
        unsigned detector_version = 7;

        /** \brief Confidence threshold.
         *
         * Higher values mean greater confidence and will result in fewer
         * faces being detected.
         *
         * For all detectors, 0.0f is the recommended default threshold.
         * A value of -0.5 usually gives a few more true faces with minimum
         * additional false faces.
         */
        float confidence_threshold = 0.0f;

        /** \brief Landmark detection settings.
         */
        landmark_settings landmark_detection;

        /** \brief Size of faces to search for.
         *
         * This setting determines what size (area) a large image will be
         * scaled down to before performing complete face detection.
         * The specific area in square pixels is determined by multiplying
         * this setting by a detector specific constant.
         *
         * Note that only images larger than the calculated area are scaled
         * down.  Since images are never scaled up, there is, for each
         * specific image, a certain threshold beyond which larger values
         * of this setting have no further effect.
         *
         * The time required for face detection will be linear in this value
         * since this time is generally linear in the number of pixels the
         * face detector has to work through.
         * Note again that for each individual image, there is a limit beyond
         * which larger values of this setting have no further effect on
         * detection time.  This is because the image will be passed through
         * without scaling.
         *
         * For the v6 and v7 detectors, the constant involved is
         * 589824 = 768x768.  This is considered to be the largest size
         * a face can have and still be reliably detected.  Therefore
         * a setting of 1.0 is recommended to find all large faces and
         * as many smaller faces as can be found without missing large faces.
         * Larger values of the setting may be used to find really small
         * faces in large images, but may result in larger faces being missed.
         * Smaller values of the setting may be used to speed up face detection
         * in cases where the detection of small faces is not required.
         */
        float size_range = 1.0f;

        /** \brief For v3 (fhog) detector only.
         *
         * Valid values are:<ul>
         *   <li>0: no limiting</li>
         *   <li>1: limit yaw range</li>
         *   <li>2: limit roll range</li>
         *   <li>3: limit both</li>
         * </ul>
         *
         * With the v3 detector, values > 0 will reduce detection time but
         * may result in faces which are rolled away from horizontal and/or
         * having significant yaw not being found.
         */
        unsigned v3_limit_pose = 0;

        /** \brief Method selection for image downscaling.
         *
         * Valid values are:<ul>
         *   <li>0: area/averaging method</li>
         *   <li>1: nearest neighbour method</li>
         * </ul>
         *
         * This option shouldn't have much effect on accuracy.  
         * Only speed when processing large images that must be downscaled.
         * Nearest neighbour should be a lot faster than averaging as no
         * computation is involved.
         */
        unsigned fast_scaling = 0;


        /** \brief Default settings.
         */
        constexpr detection_settings() = default;

        /** \brief Read settings from json object.
         *
         * Settings are either directly in object or in nested object: <pre>
             {
               "detection": {
                 ... settings ...
               },
             } </pre>
         *
         * "roll_range" and "yaw_range" maybe used to set both large and small.
         *
         * Landmarks example: <pre>
             "landmark_detection": {
                 "landmarks": ["dlib5","dlib68"],
                 "contrast_correction": 1,
             } </pre>
         * or: <pre>
             "landmark_detection": ["dlib5","dlib68"] </pre>
         * "eye_detection_variant" as in pca library is another alternative.
         *
         * This method throws an exception if any setting is missing or
         * invalid.
         * It will ignore any extra unrelated values.
         *
         * This constructor simply calls the assign() method after default
         * construction.
         */
        detection_settings(const json::value& obj);

        /** \brief Overwrite settings from json object.
         *
         * Same as the constructor, this method will throw an exception if
         * any settings is missing or invalid.
         */
        void assign(const json::object& obj);

        /** \brief Replace some settings from json object.
         *
         * This method will throw an exception if a non-null value is invalid,
         * but it will not throw due to a missing setting.
         */
        void amend(const json::object& obj);
    };

    /** \brief Encode settings as json object.
     */
    json::value to_json(const detection_settings& settings);
}


