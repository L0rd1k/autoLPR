#pragma once

#include "opencv2/tracking.hpp"

namespace alpr {

enum class TrackerType {
    BOOSTING = 0,
    KCF = 1,
    MEDIANFLOW = 2,
};

class Tracker {
public:
    Tracker();
    ~Tracker();
    bool create(const alpr::TrackerType type = alpr::TrackerType::MEDIANFLOW);
    bool init(cv::Mat& frame, const cv::Rect2d& bbox);
    bool update(cv::Mat& frame, cv::Rect2d& bbox);

private:
    cv::Ptr<cv::Tracker> tracker_;
    cv::Rect bbox_;
};

}  // namespace alpr