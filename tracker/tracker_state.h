#pragma once

#include "opencv2/opencv.hpp"
#include "deepsort/tracker_kalman.h"
#include "deepsort/feature_bundle.h"

namespace alpr {

class TrackerData {
public:
    TrackerKalman kalman;
    FeatureBundle feature;
};

class TrackerState {
public:
    int id;
    cv::Rect2f bbox;
};

}  // namespace alpr 