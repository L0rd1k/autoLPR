#pragma once

#include "opencv2/opencv.hpp"

namespace alpr {

class TrackerData {

};

class TrackerState {
public:
    uint32_t id;
    cv::Rect2f bbox;
};

enum class KalmanState {
    Unconfirmed = 0,
    Confirmed = 1,
    Removed = 2
};

}  // namespace alpr 