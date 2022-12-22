#pragma once

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

#include "detector_status.h"
#include "utils/img_processing.h"

namespace alpr {

class CPUYoloDetector {
public:
    CPUYoloDetector(std::string cfgFile, std::string weightsFile);
    virtual ~CPUYoloDetector();
    bool init();
    bool predict(const cv::Mat& img, std::vector<alpr::PredictedBox>& predictions);

    void setModelCfgPath(const std::string file);
    const std::string getModelCfgPath();
    void setWeightPath(const std::string file);
    const std::string setWeightPath();

private:
    std::string cfgFile_;
    std::string weightsFile_;
    
protected:
    bool isExist(const std::string& path);
    std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
    std::vector<alpr::PredictedBox> postprocess(const std::vector<cv::Mat>& outs, const cv::Size size);
    float confThreshold_ = 0.1;
    float nmsThreshold_ = 0.1;
    int inpWidth_ = 416;
    int inpHeight_ = 416;
    std::vector<cv::String> outLayerNames_;
    cv::dnn::Net net_;
};

}  // namespace alpr
