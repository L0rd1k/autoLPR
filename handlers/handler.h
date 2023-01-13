#pragma once

#include <future>
#include <memory>
#include <thread>

#include "detector/yolo/yolo_detector.h"
#include "filtering/kalman/kalman_rect_tracker.h"
#include "prediction_status.h"
#include "recognizer/recognizer.h"
#include "tracker/single_tracker.h"
#include "tracker/multi_tracker.h"

#include "utils/img_processing.h"

#include "opencv2/opencv.hpp"

namespace alpr {

class Handler {
public:
    Handler();
    bool process(cv::Mat& img);

    std::shared_ptr<std::future<alpr::PredictionStatus> > _detectorResult;
    // alpr::PredictionStatus _detectorResult;

    std::shared_ptr<alpr::DetectorBase> _detector;
    std::shared_ptr<alpr::SingleTracker> _tracker;
    // std::shared_ptr<alpr::MultiTracker> _tracker;
    // std::vector<cv::Rect2d> lastTrackedRect_;

    std::shared_ptr<alpr::Recognizer> _recognizer;
    std::shared_ptr<alpr::KalmanFilter> _kalman;
};

}  // namespace alpr