#pragma once

#include "tracker/tracker_state.h"

namespace alpr {

template <typename TrackData>
class TrackerManager {
public:
    explicit TrackerManager(std::vector<TrackData> &data, const std::array<int64_t, 2> &sz);

    void predict();
private:
    std::vector<TrackData> &data_;
    const cv::Rect2f bbox_;
};

}  // namespace alpr