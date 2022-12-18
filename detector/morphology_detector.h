#pragma once

#include "prediction_status.h"
#include "tracker/tracker.h"
#include "detector.h"


#include "opencv2/opencv.hpp"

#include <thread>
#include <future>
#include <memory>

namespace alpr {

class MorphologyDetector {
public:
    MorphologyDetector();
    bool process(cv::Mat& img);   

    std::shared_ptr<std::future<alpr::PredictionStatus> > _detectorResult;
    std::shared_ptr<alpr::Detector> _detector;                      
    std::shared_ptr<alpr::Tracker> _tracker;
    
    
    cv::Mat resized_img;
};

}  // namespace alpr