#pragma once

#include <memory>

#include "cpu_yolo_detector.h"
#include "prediction_status.h"

namespace alpr {

struct ImageSubstance {
    cv::Point topLeft;
    cv::Mat img;
};

class Detector {
public:
    Detector();
    Detector(const std::string yoloModelPath, const std::string yoloWeightsPathTV);
    PredictionStatus detect(const cv::Mat& frame);
private:
    std::shared_ptr<CPUYoloDetector> create(const std::string yoloModelPath, const std::string yoloWeightsPathTV);
    std::shared_ptr<CPUYoloDetector> yoloDetector_;
};

}  // namespace alpr