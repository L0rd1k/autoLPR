#include "multi_tracker.h"

bool alpr::MultiTracker::create(const int size, const alpr::TrackerType type) {
    for (uint8_t i = 0; i < size; i++) {
        switch (type) {
            case alpr::TrackerType::BOOSTING:
                trackers_.push_back(cv::TrackerBoosting::create());
                break;
            case alpr::TrackerType::KCF:
                trackers_.push_back(cv::TrackerKCF::create());
                break;
            case alpr::TrackerType::MEDIANFLOW:
                trackers_.push_back(cv::TrackerMedianFlow::create());
                break;
            default:
                return false;
        }
    }
    type_ = type;
    return true;
}

bool alpr::MultiTracker::init(cv::Mat& frame, const std::vector<cv::Rect2d> bbox) {
    if (is_inited()) {
        for(auto elem : trackers_) {
            elem->clear();
        }
        trackers_.clear();
        create(bbox.size());
    }
    if (trackers_.empty()) {
        create(bbox.size());
    }
    bbox_ = bbox;
    for (uint8_t i = 0; i < bbox.size(); i++) {
        if (!trackers_[i]->init(frame, bbox[i])) {
            return false;
        }
    }
    isInited_ = true;
    return true;
}

bool alpr::MultiTracker::update(cv::Mat& frame, std::vector<cv::Rect2d>& bbox) {
    if (trackers_.size() != bbox.size()) {
        return false;
    }
    for (uint8_t i = 0; i < trackers_.size(); i++) {
        if (trackers_[i]) {
            if (!trackers_[i]->update(frame, bbox[i])) {
                return false;
            }
        }
    }
    return true;
}
