#pragma once

#include <stdext/binary.hpp>

#include <istream>
#include <functional>
#include <string_view>
#include <variant>
#include <vector>


namespace models {

    /** \brief Format of model file.
     */
    enum class format {
        dlib,   ///< either dlib ".dat" or dlibx::net::vector ".nv"
        ncnn,   ///< two files: ".param" and ".bin"
        tflite  ///< single ".tflite" flatbuffers file
    };


    /** \brief Type of detector or classifier requiring model file.
     */
    enum class type {
        face_detector,
        landmark_detector,
        classifier,
        face_recognition
    };


    /** \brief Face detector name constants.
     *
     * Face detector names as requested by det library.
     */
    struct face_detector {
        static constexpr auto fhog = "fhog";         ///< v3 (dlib)
        static constexpr auto cnn = "cnn";           ///< v4 (dlib)
        static constexpr auto tiny = "tiny";         ///< v5 (dlib)
        static constexpr auto rfb320 = "rfb320";     ///< v6 (dlib or ncnn)
        static constexpr auto retina = "retina";     ///< v7 (dlib or ncnn)
        static constexpr auto blaze128 = "blaze128"; ///< v8 (tflite)
    };


    /** \brief Landmark detector name constants.
     *
     * Landmark detector names as requested by det library.
     */
    struct landmark_detector {
        static constexpr auto dlib5 = "dlib5";
        static constexpr auto dlib68 = "dlib68";
        static constexpr auto mesh68 = "mesh68"; // subset of mesh478
        static constexpr auto mesh478 = "mesh478"; // MediaPipe FaceMesh
    };


    /** \brief Map face recognition version number to name.
     *
     * \returns "recVER" where VER is the supplied version number
     */
    inline auto face_recognition(unsigned ver) {
        return std::string("rec") + std::to_string(ver);
    }


    /** \brief Any istream object -- typically an open file.
     *
     * \sa open_binary_file() to create one of these from a path
     */
    using istream_ptr = std::unique_ptr<std::istream>;


    /** \brief Return type for model loader method.
     */
    struct loader_return_type {
        /** \brief One or more open files or binary (serialized) data.
         *
         * Note that some implementations (e.g. ncnn) require 2 files so
         * that is why this is a vector.
         */
        std::vector<std::variant<istream_ptr,stdx::binary> > models;

        /** \brief Diagnostic path.
         *
         * This value is for diagnostic purposes and may be left empty.
         * The loaders that load files from a models directory will set
         * this value to the generic_string() value of the model file's
         * canonical path.
         */
        std::string path;

        /** \brief Default construct empty object.
         */
        loader_return_type() = default;

        /** \brief Construct from one or more open files or binaries.
         */
        template <typename... Args>
        loader_return_type(std::variant<istream_ptr,stdx::binary> first,
                           Args&&... others)
            : models{move(first),move(others)...} {
        }
        template <typename... Args>
        loader_return_type(std::string path,
                           std::variant<istream_ptr,stdx::binary> first,
                           Args&&... others)
            : models{move(first),move(others)...},
              path(move(path)) {
        }
    };


    /** \brief Type of function as accepted by det and rec libraries.
     *
     * Models are referenced by the tuple: format, type and a name.
     * 
     * The face detector names are as specified in struct face_detector.
     * The landmark detector names are as specified in struct landmark_detector.
     * Classifiers have their specific names (with format::dlib).
     * Recognition models are named "recVER" where VER is the version number.
     *
     * Models may be provided by three methods:
     *  1. [std::istream] open file on disk
     *  2. [stdx::binary] memory mapped file on disk
     *  3. [stdx::binary] encoded model exists in memory (e.g. an asset)
     *
     * The above list is in order from most efficient to least efficient, 
     * except in the case where the model is embedded in the executable as
     * an asset.  In that case option 3 is the only choice that makes sense.
     *
     * Note that loading the model from disk into memory and then having it
     * deserialized is inherently inefficient due to the excess memory
     * consumption and unnecessary copying of data.
     */
    using loader_function =
        std::function<loader_return_type(format,type,std::string_view)>;
}

