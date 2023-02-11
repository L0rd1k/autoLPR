#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "hungarian.h"
#include "tracker/tracker_state.h"
#include "utils/size.h"

namespace alpr {

using d_metrics = std::function<torch::Tensor(const std::vector<int> id_tracks, const std::vector<int> id_detects)>;
constexpr float INVALID_DIST = 1E3f;

static void combine_prediction_with_id(const d_metrics &metric,
                                       std::vector<int> &unmatched_trackers,
                                       std::vector<int> &unmatched_pred,
                                       std::vector<std::tuple<int, int>> &matched) {
    auto dist = metric(unmatched_trackers, unmatched_pred);
    auto dist_acc = dist.accessor<float, 2>();
    auto dist_vec = std::vector<std::vector<double>>(
        dist.size(0),
        std::vector<double>(dist.size(1)));
    for (size_t i = 0; i < dist.size(0); ++i) {
        for (size_t j = 0; j < dist.size(1); ++j) {
            dist_vec[i][j] = dist_acc[i][j];
        }
    }

    std::vector<int> assignment;
    Hungarian().run(dist_vec, assignment);
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] == -1) {
            continue;
        }
        if (dist_vec[i][assignment[i]] > INVALID_DIST / 10) {
            assignment[i] = -1;
        } else {
            matched.emplace_back(std::make_tuple(unmatched_trackers[i], unmatched_pred[assignment[i]]));
        }
    }
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] != -1) {
            unmatched_trackers[i] = -1;
        }
    }

    unmatched_trackers.erase(std::remove_if(unmatched_trackers.begin(), unmatched_trackers.end(),
                                            [](int i) { return i == -1; }),
                             unmatched_trackers.end());
    std::sort(assignment.begin(), assignment.end());
    std::vector<int> unmatched_pred_new;
    std::set_difference(unmatched_pred.begin(), unmatched_pred.end(),
                        assignment.begin(), assignment.end(),
                        std::inserter(unmatched_pred_new, unmatched_pred_new.begin()));
    unmatched_pred = std::move(unmatched_pred_new);
}

template <typename TrackData>
class TrackerManager {
public:
    TrackerManager(std::vector<TrackData> &data, alpr::Size<int64_t> &sz)
        : data_(data),
          bbox_(0, 0, sz.getHeight(), sz.getWidth()) {
        std::cout << ">   Cnctr: Tracker manager: " << data.size() << std::endl;
    }

    /** @brief Predict bbox data with Kalman filter. **/
    void predict() {
        for (auto &filter : data_) {
            filter.kalman.predict();
        }
    }

    /** @brief Remove empty bboxes after filtering. **/
    void remove_nan() {
        data_.erase(std::remove_if(data_.begin(), data_.end(),
                                   [](const TrackData &t_data) {
                                       auto bbox = t_data.kalman.getBox();
                                       return std::isnan(bbox.x) || std::isnan(bbox.y) || std::isnan(bbox.width) || std::isnan(bbox.height);
                                   }),
                    data_.end());
    }

    void remove() {
        data_.erase(std::remove_if(data_.begin(), data_.end(),
                                   [this](const TrackData &t_data) {
                                       return t_data.kalman.getState() == alpr::KalmanState::Removed;
                                   }),
                    data_.end());
    }

    std::vector<std::tuple<int, int>> update(const std::vector<cv::Rect2f> &pred,
                                             const d_metrics &confirmed_metrics,
                                             const d_metrics &unconfirmed_metrics) {
        std::vector<int> unmatched_trackers;
        std::cout << "Manager: data size = " << data_.size() << std::endl;
        for (size_t i = 0; i < data_.size(); ++i) {
            if (data_[i].kalman.getState() == alpr::KalmanState::Confirmed) {
                unmatched_trackers.emplace_back(i);
            }
        }
        
        std::vector<int> unmatched_pred(pred.size());
        std::iota(unmatched_pred.begin(), unmatched_pred.end(), 0);

        std::vector<std::tuple<int, int>> matched;
        combine_prediction_with_id(confirmed_metrics, unmatched_trackers, unmatched_pred, matched);
        for (size_t i = 0; i < data_.size(); ++i) {
            if (data_[i].kalman.getState() == alpr::KalmanState::Unconfirmed) {
                unmatched_trackers.emplace_back(i);
            }
        }
        combine_prediction_with_id(unconfirmed_metrics, unmatched_trackers, unmatched_pred, matched);
        for (auto i : unmatched_trackers) {
            data_[i].kalman.miss();
        }
        for (auto [x, y] : matched) {
            data_[x].kalman.update(pred[y]);
        }
        for (auto elem : unmatched_pred) {
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
            auto bbox = track.kalman.getBox();
            if (track.kalman.getState() == alpr::KalmanState::Confirmed &&
                bbox_.contains(bbox.tl()) && bbox_.contains(bbox.br())) {
                result.push_back(alpr::TrackerState{track.kalman.getId(), bbox});
            }
        }
        return result;
    }

private:
    std::vector<TrackData> &data_;
    const cv::Rect2f bbox_;
};

}  // namespace alpr