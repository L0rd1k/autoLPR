#include "gpu_yolo_detector.h"

alpr::GPUYoloDetector::GPUYoloDetector(std::string cfgFile, std::string weightsFile)
    : cfgFile_(cfgFile),
      weightsFile_(weightsFile),
      detector_(cfgFile, weightsFile) {
}

alpr::GPUYoloDetector::~GPUYoloDetector() {
}

void alpr::GPUYoloDetector::setModelCfgPath(const std::string file) {
    cfgFile_ = file;
}

const std::string alpr::GPUYoloDetector::getModelCfgPath() {
    return cfgFile_;
}

void alpr::GPUYoloDetector::setWeightPath(const std::string file) {
    weightsFile_ = file;
}

const std::string alpr::GPUYoloDetector::setWeightPath() {
    return weightsFile_;
}

bool alpr::GPUYoloDetector::predict(const cv::Mat& img, std::vector<alpr::PredictedBox>& predictions, float treshold) {
    std::vector<bbox_t> results = detector_.detect(img, treshold, true);
    float confidence = 0.0;
    for (const auto& result : results) {
        if (result.prob > confidence) {
            confidence = result.prob;
            predictions.push_back(alpr::PredictedBox(
                result.obj_id, result.prob,
                cv::Rect(result.x, result.y, result.w, result.h)));
        }
    }
    return true;
}