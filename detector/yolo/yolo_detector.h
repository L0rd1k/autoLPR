#pragma once

#include <memory>

#include "cpu_yolo_detector.h"
#include "gpu_yolo_detector.h"

#include "detector_base.h"

namespace alpr {

class YoloDetector : public alpr::DetectorBase {
public:
    YoloDetector();
    PredictionStatus detect(const cv::Mat& frame) override;
    // std::shared_ptr<CPUYoloDetector> create(const std::string yoloModelPath, const std::string yoloWeightsPathTV);
    std::shared_ptr<GPUYoloDetector> create(const std::string yoloModelPath, const std::string yoloWeightsPathTV);

private:
    // std::shared_ptr<CPUYoloDetector> yoloDetector_;
    std::shared_ptr<GPUYoloDetector> yoloDetector_;

};

}  // namespace alpr