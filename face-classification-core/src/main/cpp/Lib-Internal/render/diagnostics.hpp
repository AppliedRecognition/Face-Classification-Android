#pragma once

#include <opencv2/core/core.hpp>

namespace render {
    struct diagnostics {
        cv::Mat before_lighting;
        float lighting_weight;
        std::vector<cv::Point> final_landmarks;
    };
}
