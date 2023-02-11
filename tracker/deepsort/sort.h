#pragma once

#include <array>
#include <vector>
#include <memory>
#include <stdint.h>

#include "tracker/tracker_state.h"
#include "tracker_manager.h"
#include "nn_matching.h"
#include "utils/size.h"

#include <opencv2/opencv.hpp>

namespace alpr {

template<typename TrackData>
class TrackerManager;
class TrackerData;

class Sort {
public:
    explicit Sort(alpr::Size<int64_t> &sz);
    ~Sort();
    std::vector<alpr::TrackerState> update(const std::vector<cv::Rect2f> &pred);
private:
    using Manager = alpr::TrackerManager<alpr::TrackerData>;
    std::vector<alpr::TrackerData> data_;
    std::unique_ptr<Manager> manager_;

};

}  // namespace alpr