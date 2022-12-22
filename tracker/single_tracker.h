#pragma once

#include "tracker.h"

namespace alpr {

class SingleTracker : public alpr::Tracker {
public:
    bool create(const alpr::TrackerType type = alpr::TrackerType::MEDIANFLOW);
    bool init(cv::Mat& frame, const cv::Rect2d bbox);
    bool update(cv::Mat& frame, cv::Rect2d& bbox);
private:
    cv::Ptr<cv::Tracker> tracker_;
    cv::Rect bbox_;
    alpr::TrackerType type_;
};

}  // namespace alpr