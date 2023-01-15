#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <numeric>

#include <torch/torch.h>

#include "tracker/tracker_state.h"
#include "hungarian.h"

namespace alpr {


using d_metrics = std::function<torch::Tensor(const std::vector<int> id_tracks, const std::vector<int> id_detects)>;

static void combine_prediction_with_id(const d_metrics &metric, 
                                std::vector<int> &unmatched_trackers, 
                                std::vector<int> &unmatched_pred,
                                std::vector<std::tuple<int, int>> &matched) {
    auto dist = metric(unmatched_trackers, unmatched_pred);
    auto dist_acc = dist.accessor<float, 2>();
    auto dist_vec = std::vector<std::vector<double>>(
                        dist.size(0),
                        std::vector<double>(dist.size(1)));
    for(size_t i = 0; i < dist.size(0); ++i) {
        for(size_t j = 0; j < dist.size(1); ++j) {
            dist_vec[i][j] = dist_acc[i][j];
        }
    }
    std::vector<int> assignment;




}

template <typename TrackData>
class TrackerManager {
public:
    TrackerManager(std::vector<TrackData> &data, const std::array<int64_t, 2> &sz)
        : data_(data),
          bbox_(0, 0, sz[1], sz[0]) {
    }

    void predict() {
        for (const auto &filter : data_) {
            filter.kalman.predict();
        }
    }

    void remove_nan() {
        data_.erase(std::remove_if(data_.begin(), data_.end(),
            [](const TrackData &t_data) {
                auto bbox = t_data.kalman.rect();
                return std::isnan(bbox.x) || std::isnan(bbox.y) || std::isnan(bbox.width) || std::isnan(bbox.height);
            }), data_.end());
    }

    void remove() {
        data_.erase(std::remove_if(data_.begin(), data_.end(), 
         [this](const TrackData &t_data) {
             return t_data.kalman.getState() = alpr::KalmanState::Removed;
         }), data_.end());
    }

    std::vector<std::tuple<int, int>> update(const std::vector<cv::Rect2f> &pred,
                                             const d_metrics &confirmed_metrics, 
                                             const d_metrics &unconfirmed_metrics) {
        std::vector<int> unmatched_trackers;
        for (size_t i = 0; i < data_.size(); ++i) {
            if(data_[i].kalman.getState() == alpr::KalmanState::Confirmed) {
                unmatched_trackers.emplace_back(i);
            }
        }

        std::vector<int> unmatched_pred(pred.size());
        std::iota(unmatched_pred.begin(), unmatched_pred.end(), 0);

        std::vector<std::tuple<int, int>> matched;
        combine_prediction_with_id(confirmed_metrics, unmatched_trackers, unmatched_pred, matched);
        for(size_t i = 0; i < data_.size(); ++i) {
            if(data_[i].kalman.getState() == alpr::KalmanState::Unconfirmed) {
                unmatched_trackers.emplace_back(i);
            }
        }
        combine_prediction_with_id(unconfirmed_metrics, unmatched_trackers, unmatched_pred, matched);
        for (auto i : unmatched_trackers) {
            data_[i].kalman.miss();
        }
        for(auto[x, y] : matched) {
            data_[x].kalman.update(pred[y]);
        }
        for(auto elem : unmatched_pred) {
            matched.emplace_back(data_.size(), elem);
            auto td = TrackData{};
            td.kalman.init(pred[elem]);
            data_.emplace_back(td);
        }
        return matched;
    }
    
    
    std::vector<alpr::TrackerState> visible_tracks() {
        std::vector<alpr::TrackerState> result;
        for (auto &track : data_) {
            auto bbox = track.kalman.rect();
            if(track.kalman.getState() == alpr::KalmanState::Confirmed && 
                bbox_.contains(bbox.tl()) && bbox_.contains(bbox.br())) {
                result.push_back(alpr::TrackerState{track.kalman.gedId(), bbox});
            }
        }
        return result;
    }

private:
    std::vector<TrackData> &data_;
    const cv::Rect2f bbox_;
};

}  // namespace alpr