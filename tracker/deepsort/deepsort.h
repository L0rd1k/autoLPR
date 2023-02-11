#pragma once

#include <stdint.h>

#include <array>
#include <memory>
#include <vector>

#include "extractor.h"
#include "feature_bundle.h"
#include "feature_metric.h"
#include "nn_matching.h"
#include "tracker/tracker_state.h"
#include "tracker_manager.h"
#include "utils/size.h"

namespace alpr {

template <typename TrackData>
class TrackerManager;
class TrackerData;

class DeepSort {
public:
    explicit DeepSort(alpr::Size<int64_t> &sz);
    ~DeepSort();
    std::vector<alpr::TrackerState> update(const std::vector<cv::Rect2f> &pred, const cv::Mat &img);

private:
    using Manager = alpr::TrackerManager<alpr::TrackerData>;
    using FeatMetric = alpr::FeatureMetric<alpr::TrackerData>;
    std::vector<alpr::TrackerData> data_;  //> Vector of tracked objects
    std::unique_ptr<Manager> manager_;
    std::unique_ptr<FeatMetric> fmetric_;
    std::unique_ptr<Extractor> extractor_;
};

}  // namespace alpr