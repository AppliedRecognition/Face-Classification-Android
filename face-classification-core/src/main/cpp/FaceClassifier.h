// FaceClassifier.h
#pragma once

#include <string>
#include <vector>
#include <core/context.hpp>
#include <det/types.hpp>
#include <det/classifiers.hpp>

namespace verid {
    class FaceClassifier {
    public:
        // Constructor
        FaceClassifier(const std::string &name, std::unique_ptr<stdx::binary> modelBuffer);

        // Main method as per requirements
        std::vector<float> extractAttribute(const det::face_coordinates &face, const raw_image::plane &image);

        // (Destructor and copy/move operations can be defaulted or deleted as needed)
    private:
        core::context_ptr context;
        det::classifier_model_type const* classifier;
    };
}