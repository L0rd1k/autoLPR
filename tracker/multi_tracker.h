#pragma once

#include "tracker.h"

namespace alpr {

class MultiTracker : public alpr::Tracker {
public:
    bool create(const int size, const alpr::TrackerType type = alpr::TrackerType::MEDIANFLOW);
    bool init(cv::Mat& frame, const std::vector<cv::Rect2d> bbox);
    bool update(cv::Mat& frame, std::vector<cv::Rect2d>& bbox);
private:
    std::vector<cv::Rect2d> bbox_;
    std::vector<cv::Ptr<cv::Tracker>> trackers_;
    alpr::TrackerType type_;
};

}  // namespace alpr