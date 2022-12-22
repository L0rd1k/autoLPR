#pragma once

#include <memory>

#include "prediction_status.h"

namespace alpr {

struct ImageSubstance {
    cv::Point topLeft;
    cv::Mat img;
};

class DetectorBase {
public:
    virtual PredictionStatus detect(const cv::Mat& frame) = 0;
};

}  // namespace alpr