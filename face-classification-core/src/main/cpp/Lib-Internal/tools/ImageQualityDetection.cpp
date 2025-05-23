//
//  ImageQualityDetection.cpp
//  VerIDCore
//
//  Created by Jakub Dolejs on 01/02/2019.
//  Copyright © 2019 Applied Recognition. All rights reserved.
//

#include "ImageQualityDetection.hpp"

#include <opencv2/imgproc/imgproc.hpp>

double verid::sharpnessOfImage(
    const void* buffer, unsigned width, unsigned height) {
    const auto image =
        cv::Mat(int(height), int(width), CV_8UC1, const_cast<void*>(buffer));
    static constexpr auto maxSize = 320;
    const auto scale = maxSize / double(std::min(width,height));
    cv::Mat resized;
    cv::resize(image, resized, {}, scale, scale, cv::INTER_CUBIC);
    cv::Mat laplacian;
    cv::Laplacian(resized, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return stddev[0] * stddev[0];
}

verid::bcs_result
verid::brightnessContrastAndSharpnessOfImage(
    const void* buffer, unsigned width, unsigned height) {
    const auto image =
        cv::Mat(int(height), int(width), CV_8UC1, const_cast<void*>(buffer));
    static constexpr auto maxSize = 320;
    const auto scale = maxSize / double(std::min(width,height));
    cv::Mat resized;
    cv::resize(image, resized, {}, scale, scale, cv::INTER_CUBIC);

    cv::Scalar mean, stddev;
    cv::meanStdDev(resized, mean, stddev);
    const auto brightness = mean[0];
    const auto contrast = stddev[0];

    cv::Mat laplacian;
    cv::Laplacian(resized, laplacian, CV_64F);
    cv::meanStdDev(laplacian, mean, stddev);
    const auto sharpness = 100 * stddev[0] / std::max(contrast,1.0);

    const auto middle = resized.cols/2;
    const auto mean_left  = cv::mean(resized.colRange(0,middle));
    const auto mean_right = cv::mean(resized.colRange(middle,resized.cols));
    const auto horz = mean_left[0] - mean_right[0];

    return { brightness, contrast, sharpness, horz, 0 };
}
