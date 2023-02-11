#pragma once

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "detector/yolo/yolo_detector.h"
#include "filtering/kalman/kalman_rect_tracker.h"
#include "prediction_status.h"
#include "recognizer/recognizer.h"
#include "tracker/multi_tracker.h"
#include "tracker/single_tracker.h"
#include "tracker/deepsort/deepsort.h"
#include "utils/img_processing.h"
#include "utils/size.h"
#include "utils/ring_queue.h"

namespace alpr {

class Handler {
public:
    Handler();
    bool process(cv::Mat& img);

    std::shared_ptr<std::future<alpr::PredictionStatus> > _detectorResult;
    std::shared_ptr<alpr::DetectorBase> _detector;
    std::shared_ptr<alpr::SingleTracker> _tracker;
    std::shared_ptr<alpr::Recognizer> _recognizer;
    std::shared_ptr<alpr::KalmanFilter> _kalman;
    std::shared_ptr<alpr::DeepSort> _deepTracker;
};

}  // namespace alpr