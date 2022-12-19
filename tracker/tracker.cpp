#include "tracker.h"

alpr::Tracker::Tracker() {
}

alpr::Tracker::~Tracker() {
}

bool alpr::Tracker::create(const alpr::TrackerType type) {
    switch (type) {
        case alpr::TrackerType::BOOSTING:
            tracker_ = cv::TrackerBoosting::create();
        case alpr::TrackerType::KCF:
            tracker_ = cv::TrackerKCF::create();
        case alpr::TrackerType::MEDIANFLOW:
            tracker_ = cv::TrackerMedianFlow::create();
        default:
            return false;
    }
    type_ = type;
    return true;
}

bool alpr::Tracker::init(cv::Mat& frame, const cv::Rect2d bbox) {
    if (is_inited()) {
        tracker_->clear();
        create(type_);
    }
    bbox_ = bbox;
    if (tracker_->init(frame, bbox_)) {
        isInited_ = true;
        return true;
    }
    return false;
}

bool alpr::Tracker::is_inited() {
    return isInited_;
}

void alpr::Tracker::set_inited(bool flag) {
    isInited_ = flag;
}

bool alpr::Tracker::update(cv::Mat& frame, cv::Rect2d& bbox) {
    if (tracker_) {
        return tracker_->update(frame, bbox);
    }
    return false;
}