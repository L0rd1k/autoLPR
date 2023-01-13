#pragma once

#include "opencv2/tracking.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/tracking/tracking.hpp"

namespace alpr {

enum class TrackerType {
    BOOSTING = 0,
    KCF = 1,
    MEDIANFLOW = 2,
    DEEPSORT = 3,
};

class Tracker {
public:
    bool is_inited();
    void set_inited(bool flag);
protected:
    bool isInited_ = false;
};

}  // namespace alpr