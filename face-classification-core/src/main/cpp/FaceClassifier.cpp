#include "FaceClassifier.h"
#include "conversion.h"
#include <memory>
#include <filesystem>
#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include <fstream>
#include <det_dlib/init.hpp>
#include <det/image.hpp>
#include <det/landmark_standardize.hpp>
#include <det/classifiers.hpp>
#include <det/detection_settings.hpp>
#include <det/detection.hpp>
#include <det/types.hpp>
#include <render/frontalize.hpp>
#include <core/context.hpp>
#include <raw_image/transform.hpp>
#include <raw_image/point_rounding.hpp>
#include <core/thread_data.hpp>
#include <stdext/base64.hpp>
#include <json/types.hpp>
#include <csignal>
#include <cstdlib>
#include <execinfo.h>
#include <stdext/span.hpp>

static auto contextSettings() {
    core::context_settings cs;
    cs.max_threads = 8;  // auto detect
    return cs;
}

namespace verid {

    FaceClassifier::FaceClassifier(const std::string &name, std::unique_ptr<stdx::binary> modelBuffer) : context(
            core::context::construct(contextSettings())) {
        if (!modelBuffer || modelBuffer->empty()) {
            throw std::runtime_error("Failed to read model buffer for: " + name);
        }
        classifier = det::load_classifier(context, name, *modelBuffer, name);
        if (!classifier) {
            throw std::runtime_error("Failed to initialize classifier");
        }
    }

    std::vector<float> FaceClassifier::extractAttribute(const det::face_coordinates &face, const raw_image::plane &image) {
        std::vector<float> result = det::apply_classifier(context, classifier, image, face);
        return result;
    }
}