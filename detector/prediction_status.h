#pragma once

#include <opencv2/core.hpp>

namespace alpr {

struct PredictionStatus {
    PredictionStatus()
        : found(false) {
    }
    bool found;
    std::vector<cv::Rect> rects;
};

}  // namespace alpr