#pragma once

#include <string>

#include "yolo_v2_class.hpp"
#include "detector_status.h"

namespace alpr {

class GPUYoloDetector {
public:
    GPUYoloDetector(std::string cfgFile, std::string weightsFile);
    virtual ~GPUYoloDetector();
    bool predict(const cv::Mat& img, std::vector<alpr::PredictedBox>& predictions, float treshold = 0.1);

    void setModelCfgPath(const std::string file);
    const std::string getModelCfgPath();
    void setWeightPath(const std::string file);
    const std::string setWeightPath();

private:
    std::string cfgFile_;
    std::string weightsFile_;
    Detector detector_;

};

}  // namespace alpr
