#pragma once

#include "prediction_status.h"
#include "tracker/tracker.h"
#include "recognizer/recognizer.h"
#include "detector.h"
#include "filtering/kalman/kalman_rect_tracker.h"

#include "opencv2/opencv.hpp"

#include <thread>
#include <future>
#include <memory>

namespace alpr {

class Handler {
public:
    Handler();
    bool process(cv::Mat& img);   

    std::shared_ptr<std::future<alpr::PredictionStatus> > _detectorResult;
    std::shared_ptr<alpr::Detector> _detector;                      
    std::shared_ptr<alpr::Tracker> _tracker;
    std::shared_ptr<alpr::Recognizer> _recognizer;
    std::shared_ptr<alpr::KalmanFilter> _kalman;
};

}  // namespace alpr