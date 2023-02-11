#include "deepsort.h"

alpr::DeepSort::DeepSort(alpr::Size<int64_t> &sz)
    : manager_(std::make_unique<Manager>(data_, sz)),
      fmetric_(std::make_unique<FeatMetric>(data_)),
      extractor_(std::make_unique<Extractor>("weights/ckpt.bin")) {
}

alpr::DeepSort::~DeepSort() {
}

std::vector<alpr::TrackerState> alpr::DeepSort::update(const std::vector<cv::Rect2f> &pred, const cv::Mat &img) {
    manager_->predict();
    manager_->remove_nan();
    auto confirmed_metrics = [this, &pred, &img](const std::vector<int> &tracks_id, const std::vector<int> &pred_id) {
        std::vector<cv::Rect2f> trackers, preds;
        std::vector<cv::Mat> boxes;
        // List of bboxs after kalman filtering.
        for (auto id : tracks_id) {
            trackers.push_back(data_[id].kalman.getBox());
        }
        //> Copy predicted bboxes templates.
        for (auto id : pred_id) {
            preds.push_back(pred[id]);
            boxes.push_back(img(pred[id]));
        }
        // Find intersection between predicated and filtered bboxes.
        torch::Tensor iou_mat = iou_distance(preds, trackers);
        auto feature_mat = fmetric_->distance(extractor_->extract(boxes), tracks_id);
        feature_mat.masked_fill_((iou_mat > 0.8f).__ior__(feature_mat > 0.2f), INVALID_DIST);
        return feature_mat;
    };

    auto unconfirmed_metrics = [this, &pred](const std::vector<int> &tracks_id, const std::vector<int> &pred_id) {
        std::vector<cv::Rect2f> trackers;
        for (auto id : tracks_id) {
            trackers.push_back(data_[id].kalman.getBox());
        }
        std::vector<cv::Rect2f> preds;
        for (auto id : pred_id) {
            preds.push_back(pred[id]);
        }
        torch::Tensor iou_mat = iou_distance(preds, trackers);
        iou_mat.masked_fill_(iou_mat > 0.7f, INVALID_DIST);
        return iou_mat;
    };

    auto matched = manager_->update(pred, confirmed_metrics, unconfirmed_metrics);
    std::vector<cv::Mat> boxes;
    std::vector<int> targets;
    for (auto [x, y] : matched) {
        targets.emplace_back(x);
        boxes.emplace_back(img(pred[y]));
    }
    fmetric_->update(extractor_->extract(boxes), targets);
    manager_->remove();
    return manager_->visible_tracks();
}
