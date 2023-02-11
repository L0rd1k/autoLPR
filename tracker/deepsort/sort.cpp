#include "sort.h"

alpr::Sort::Sort(alpr::Size<int64_t> &sz)
    : manager_(std::make_unique<Manager>(data_, sz)) {
}

alpr::Sort::~Sort() {
}

std::vector<alpr::TrackerState> alpr::Sort::update(const std::vector<cv::Rect2f> &pred) {
    manager_->predict();
    manager_->remove_nan();

    auto metric = [this, &pred](const std::vector<int> &trackers_id, const std::vector<int> &detections_id) {
        std::vector<cv::Rect2f> trackers;
        for(auto id : trackers_id) {
            trackers.push_back(data_[id].kalman.getBox());
        }
        std::vector<cv::Rect2f> predictions;
        for(auto &id : detections_id) {
            predictions.push_back(pred[id]);
        } 
        auto iou_mat = alpr::iou_distance(predictions, trackers);
        iou_mat.masked_fill_(iou_mat > 0.7f, INVALID_DIST);
        return iou_mat;
    };
    manager_->update(pred, metric, metric);
    manager_->remove();
    return manager_->visible_tracks();
}