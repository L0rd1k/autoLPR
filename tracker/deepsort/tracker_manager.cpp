#include "tracker_manager.h"

template <typename TrackData>
alpr::TrackerManager<TrackData>::TrackerManager(std::vector<TrackData> &data, const std::array<int64_t, 2> &sz)
    : data_(data),
      bbox_(0, 0, sz[1], sz[0]) {
}

template <typename TrackData>
void alpr::TrackerManager<TrackData>::predict() {
    for(const auto& track : data_ ) {
        track.kalman.predict();
    }
}
