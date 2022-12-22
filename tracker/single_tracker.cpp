#include "single_tracker.h"

bool alpr::SingleTracker::create(const alpr::TrackerType type) {
    switch (type) {
        case alpr::TrackerType::BOOSTING:
            tracker_ = cv::TrackerBoosting::create();
            break;
        case alpr::TrackerType::KCF:
            tracker_ = cv::TrackerKCF::create();
            break;
        case alpr::TrackerType::MEDIANFLOW:
            tracker_ = cv::TrackerMedianFlow::create();
            break;
        default:
            return false;
    }
    type_ = type;
    return true;
}

bool alpr::SingleTracker::init(cv::Mat& frame, const cv::Rect2d bbox) {
    if (is_inited()) {
        tracker_->clear();
        create(type_);
    }
    bbox_ = bbox;  
    
    if(!tracker_) {
        create(alpr::TrackerType::MEDIANFLOW);
    }
    
    if (tracker_->init(frame, bbox_)) {
        isInited_ = true;
        return true;
    }
    return false;
}

bool alpr::SingleTracker::update(cv::Mat& frame, cv::Rect2d& bbox) {
    if (tracker_) {
        return tracker_->update(frame, bbox);
    }
    return false;
}