#pragma once

#include "opencv2/video/tracking.hpp"
#include "tracker/tracker_state.h"

namespace alpr {

class TrackerKalman {
public:
    TrackerKalman();
    TrackerKalman(cv::Rect2f box);
    void init(cv::Rect2f box);
    void predict();
    void update(cv::Rect2f state_box);
    void miss();
    cv::Rect2f getBox() const;
    int getId() const;
private:
    cv::KalmanFilter kalman_;
    cv::Mat measurement_;
    int update_timer = 0;
    int hits = 0;
    int id_ = -1;
    alpr::KalmanState k_state_ = alpr::KalmanState::Unconfirmed;
    static const auto init_counts = 3;
    static const auto max_stage = 30;
    static int count;
};

}  // namespace alpr