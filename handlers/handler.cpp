#include "handler.h"

alpr::Handler::Handler() {
    _detector = std::make_shared<alpr::Detector>();
    _tracker = std::make_shared<alpr::Tracker>();
    _kalman = std::make_shared<alpr::KalmanRectTracker>(6, 4, 0);
    _tracker->create(alpr::TrackerType::MEDIANFLOW);
    _kalman->init();
}

bool alpr::Handler::process(cv::Mat& img) {
    cv::Rect2d lastTrackedRect_;

    if (_detectorResult == nullptr) {
        _detectorResult = std::make_shared<std::future<alpr::PredictionStatus>>(
            std::async(std::launch::async, &alpr::Detector::detect, _detector, img));
    }

    alpr::PredictionStatus prediction;
    if (_detectorResult->valid() == true) {
        auto state = _detectorResult->wait_for(std::chrono::milliseconds(0));
        if (state == std::future_status::ready) {
            prediction = _detectorResult->get();
            _detectorResult.reset();
        }
    }

    if(prediction.found) {
        if(_tracker->init(prediction.frame, prediction.rects[0])) {
            if(_tracker->update(img, lastTrackedRect_)) {
                lastTrackedRect_ = prediction.rects[0];
            }
        }
    } else {
        if(_tracker->is_inited()) {
            if(_tracker->update(img, lastTrackedRect_)) {
                cv::Rect newRect = _kalman->update(lastTrackedRect_);
                cv::rectangle(img, lastTrackedRect_, cv::Scalar(0, 0, 255), 3);
                cv::rectangle(img, newRect, cv::Scalar(255, 0, 0), 3);
            }
        }
    }

    if(!lastTrackedRect_.empty() && !img.empty()) {
        lastTrackedRect_ = fixBound(lastTrackedRect_, img.size());
        _recognizer->process(img, lastTrackedRect_);
    }
    
    cv::imshow("Result", img);
    cv::waitKey(1);
    return true;

}
