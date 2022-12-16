#include "tracker.h"

alpr::Tracker::Tracker() {
}

alpr::Tracker::~Tracker() {
}

bool alpr::Tracker::create(const alpr::TrackerType type) {
    if(!tracker_) {
        switch(type) {
            case alpr::TrackerType::BOOSTING:
                tracker_ = cv::TrackerBoosting::create();
            case alpr::TrackerType::KCF:
                tracker_ = cv::TrackerKCF::create();
            case alpr::TrackerType::MEDIANFLOW:
                tracker_ = cv::TrackerMedianFlow::create();
        }
        return true;
    }
    return false;
}

bool alpr::Tracker::init(cv::Mat& frame, const cv::Rect2d& bbox) {
    bbox_ = bbox;
    cv::Mat roi = frame(bbox_);
    tracker_->init(roi, bbox_);
    return true;
}

bool alpr::Tracker::update(cv::Mat& frame, cv::Rect2d& bbox) {
    cv::Mat roi = frame(bbox_);
    if(tracker_) {
        return tracker_->update(roi, bbox);
    }
    return false;
}