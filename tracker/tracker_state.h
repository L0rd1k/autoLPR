#pragma once

#include "opencv2/opencv.hpp"
#include "deepsort/tracker_kalman.h"

namespace alpr {

class TrackerData {
    TrackerKalman kalman;
};

class TrackerState {
public:
    uint32_t id;
    cv::Rect2f bbox;
};

}  // namespace alpr 