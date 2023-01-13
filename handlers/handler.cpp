#include "handler.h"

alpr::Handler::Handler() {
    _detector = std::make_shared<alpr::YoloDetector>();
    _tracker = std::make_shared<alpr::SingleTracker>();
    // _tracker = std::make_shared<alpr::MultiTracker>();

    _kalman = std::make_shared<alpr::KalmanRectTracker>(6, 4, 0);
    _tracker->create(alpr::TrackerType::MEDIANFLOW);
    _kalman->init();
}

// bool alpr::Handler::process(cv::Mat& img) {
//     std::vector<cv::Rect2d> lastTrackedRect_;

//     _detectorResult = _detector->detect(img);
//     if (_detectorResult.found) {
//         lastTrackedRect_ = _detectorResult.rects;
//         for (auto boxs : lastTrackedRect_) {
//             cv::rectangle(img, boxs, cv::Scalar(0, 0, 255), 3);
//         }
//     }

//     cv::imshow("Result", img);
//     cv::waitKey(1);
//     return true;
// }

bool alpr::Handler::process(cv::Mat& img) {
    std::vector<cv::Rect2d> lastTrackedRect_;

    if (_detectorResult == nullptr) {
        _detectorResult = std::make_shared<std::future<alpr::PredictionStatus>>(
            std::async(std::launch::async, &alpr::DetectorBase::detect, _detector, img));
    }

    alpr::PredictionStatus prediction;
    if (_detectorResult->valid() == true) {
        auto state = _detectorResult->wait_for(std::chrono::milliseconds(0));
        if (state == std::future_status::ready) {
            prediction = _detectorResult->get();
            _detectorResult.reset();
        }
    }

    if (prediction.found) {
        /*** MULTITRACK TRACKER **/
        // if(_tracker->init(prediction.frame, prediction.rects)) {
        //      if(_tracker->update(img, prediction.rects)) {
                 lastTrackedRect_ = prediction.rects;
                 for(auto boxs : lastTrackedRect_) {
                     cv::rectangle(img, boxs, cv::Scalar(0, 0, 255), 3);
                }
        //      }
        // }

        /*** SINGLE TRACKER **/
        // if (_tracker->init(prediction.frame, prediction.rects[0])) {
        //     if (_tracker->update(img, lastTrackedRect_)) {
        //         lastTrackedRect_ = prediction.rects[0];
        //         cv::rectangle(img, lastTrackedRect_, cv::Scalar(255, 255, 255), 3);
        //     }
        // }
    } else {
        /*** MULTITRACK TRACKER **/
        // if(_tracker->is_inited()) {
        //     if(_tracker->update(img, lastTrackedRect_)) {
        //          for(auto boxs : lastTrackedRect_) {
        //              cv::rectangle(img, boxs, cv::Scalar(255, 255, 255), 3);
        //          }
        //     }
        // }

        /*** SINGLE TRACKER **/
        // if (_tracker->is_inited()) {
        //     if (_tracker->update(img, lastTrackedRect_)) {
        //         cv::Rect newRect = _kalman->update(lastTrackedRect_);
        //         cv::rectangle(img, lastTrackedRect_, cv::Scalar(0, 0, 255), 3);
        //         //cv::rectangle(img, newRect, cv::Scalar(255, 0, 0), 3);
        //     }
        // }
    }

    // if (!lastTrackedRect_.empty() && !img.empty()) {
    //     lastTrackedRect_ = fixBound(lastTrackedRect_, img.size());
    //     _recognizer->process(img, lastTrackedRect_);
    // }

    cv::imshow("Result", img);
    cv::waitKey(1);
    return true;
}
