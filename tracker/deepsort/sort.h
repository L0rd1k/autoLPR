#pragma once

#include <array>
#include <vector>
#include <memory>
#include <stdint.h>

#include "tracker/tracker_state.h"
#include "tracker_manager.h"

namespace alpr {

class TrackerData;
class Sort {
public:
    explicit Sort(const std::array<uint64_t, 2> &d);
    ~Sort();
    std::vector<alpr::TrackerState> update(const std::vector<cv::Rect2f> &predictions);
private:
    using Manager = TrackerManager<alpr::TrackerData>;
    std::vector<alpr::TrackerData> data_;
    std::unique_ptr<Manager> manager_;

};

}  // namespace alpr