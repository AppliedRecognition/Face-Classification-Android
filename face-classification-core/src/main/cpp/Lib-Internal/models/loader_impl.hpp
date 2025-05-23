#pragma once

#include "loader.hpp"

#include <applog/core.hpp>

#include <algorithm>
#include <stdexcept>


/* Internal implementation details (helper methods) for class loader.
 *
 * This file should not be included outside this library.
 */

namespace models {

    /** \brief Find regular file with specified prefix and suffix.
     *
     * In the case of multiple matches, the lexicographically greatest
     * candidate is returned.
     */
    template <typename DIRECTORY_ITERATOR, typename PATH>
    auto search_for_file(const PATH& base_dir,
                         std::string_view prefix,
                         std::string_view suffix) {
        const auto plen = prefix.size();
        const auto slen = suffix.size();
        const auto min_len = plen + slen;
        std::vector<PATH> candidates;
        for (DIRECTORY_ITERATOR di(base_dir), end; di != end; ++di)
            if (is_regular_file(*di)) {
                auto&& fn = di->path().filename();
                auto&& fns = fn.generic_string();
                if (min_len <= fns.size() &&
                    fns.compare(0,plen,prefix) == 0 &&
                    fns.compare(fns.size()-slen,slen,suffix) == 0)
                    candidates.emplace_back(std::move(fn));
            }
        // choose lexicographically greatest candidate
        const auto it = std::max_element(candidates.begin(), candidates.end());
        return it != candidates.end() ? base_dir / *it : PATH{};
    }

    /// select one or more filenames for requested model
    template <typename PATH>
    template <typename DIRECTORY_ITERATOR>
    auto loader<PATH>::find_files(format f, type t, std::string_view name) const {
        const auto& mp = models_directory;
        std::vector<PATH> fns;

        switch (t) {

        case type::face_detector:
            switch (f) {
            case format::dlib:
                if (name == face_detector::fhog)
                    fns.push_back(mp / "fhogcascade_face_frontal.dat");
                else if (name == face_detector::cnn)
                    fns.push_back(mp / "mmod_human_face_detector.dat");
                else if (name == face_detector::tiny)
                    fns.push_back(mp / "faceapi_tiny_detector.dat");
                else if (name == face_detector::rfb320)
                    fns.push_back(mp / "RFB-320.nv");
                else if (name == face_detector::retina)
                    fns.push_back(mp / "mnet.25-opt.nv");
                break;

            case format::ncnn:
                if (name == face_detector::rfb320) {
                    fns.push_back(mp / "RFB-320.param");
                    fns.push_back(mp / "RFB-320.bin");
                }
                else if (name == face_detector::retina) {
                    fns.push_back(mp / "mnet.25-opt.param");
                    fns.push_back(mp / "mnet.25-opt.bin");
                }
                break;

            case format::tflite:
                if (name == face_detector::blaze128)
                    fns.push_back(mp / "blaze128.tflite");
                break;

            default:
                throw std::invalid_argument("unknown model format");
            }
            if (fns.empty())
                throw std::invalid_argument("unknown face detector");
            break;

        case type::landmark_detector:
            switch (f) {
            case format::dlib:
                if (name == landmark_detector::dlib5)
                    fns.push_back(mp / "shape_predictor_5_face_landmarks.dat");
                else if (name == landmark_detector::dlib68)
                    fns.push_back(mp / "shape_predictor_68_face_landmarks.dat");
                else if (name == landmark_detector::mesh68)
                    fns.push_back(mp / "facemesh68.nv");
                else if (name == landmark_detector::mesh478)
                    fns.push_back(mp / "facemesh478.nv");
                break;

            case format::ncnn:
                if (name == landmark_detector::mesh68) {
                    fns.push_back(mp / "facemesh68.param");
                    fns.push_back(mp / "facemesh68.bin");
                }
                else if (name == landmark_detector::mesh478) {
                    fns.push_back(mp / "facemesh478.param");
                    fns.push_back(mp / "facemesh478.bin");
                }
                break;

            case format::tflite:
                if (name == landmark_detector::mesh478)
                    fns.push_back(mp / "facemesh478.tflite");
                break;

            default:
                throw std::invalid_argument("unknown model format");
            }
            if (fns.empty())
                throw std::invalid_argument("unknown landmark detector");
            break;

        case type::classifier: {
            if (f != format::dlib)
                throw std::invalid_argument("unknown model format");
            if (name.empty())
                throw std::invalid_argument("classifier name cannot be empty");
            auto fn = search_for_file<DIRECTORY_ITERATOR>(
                mp, std::string(name) + '-', ".nv");
            if (fn.empty())
                fn = mp / (std::string(name) + ".nv");
            fns.push_back(std::move(fn));
            break;
        }

        case type::face_recognition: {
            if (name.compare(0,3,"rec") != 0)
                throw std::invalid_argument("unknown face recognition");
            if (f != format::dlib && f != format::ncnn)
                throw std::invalid_argument("unknown model format");
            const auto ext = (f == format::dlib ? ".nv" : ".param");
            PATH fn;
            if (name == "rec16" && f == format::dlib) {
                fn = mp / "dlib_face_recognition_resnet_model_v1.dat";
                if (!is_regular_file(fn))
                    fn.clear();
            }
            else if (name == "rec20") {
                fn = mp / (std::string("facenet-20170512") + ext);
                if (!is_regular_file(fn))
                    fn = search_for_file<DIRECTORY_ITERATOR>(
                        mp, "facenet-20170512-", ext);
            }
            else if (name == "rec24") {
                fn = mp / (std::string("mobilefacenet") + ext);
                if (!is_regular_file(fn))
                    fn = search_for_file<DIRECTORY_ITERATOR>(
                        mp, "mobilefacenet-", ext);
            }
            if (fn.empty()) {
                fn = search_for_file<DIRECTORY_ITERATOR>(
                    mp, std::string(name) + '-', ext);
                if (fn.empty())
                    fn = mp / (std::string(name) + ext);
            }
            fns.push_back(fn);
            if (f == format::ncnn) {
                fn.replace_extension(".bin");
                fns.push_back(fn);
            }
            break;
        }

        default:
            throw std::invalid_argument("unknown model type");
        }

        return fns;
    }

    /// open the model files from list of filenames
    template <typename PATH>
    auto loader<PATH>::open_files(const std::vector<PATH>& fns) {
        loader_return_type r;
        for (auto& fn : fns) {
            if (!is_regular_file(fn)) {
                FILE_LOG(logWARNING) << "model file not found: " << fn;
                throw std::runtime_error("model file not found: " + fn.string());
            }
            auto in = open_binary_file(fn);
            if (!in->good()) {
                FILE_LOG(logERROR) << "failed to open model file: " << fn;
                throw std::runtime_error("failed to open model file: " + fn.string());
            }
            if (r.path.empty())
                r.path = canonical(fn).generic_string();
            r.models.push_back(move(in));
        }
        return r;
    }
}
