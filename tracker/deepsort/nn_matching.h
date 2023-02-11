#pragma once

#include <torch/torch.h>

namespace alpr {

/** @brief Find the intersection distances between predicted and filtered bboxes.
 * @param bboxFiltered Filtered bounding box.
 * @param bboxPredicted Predicted bounding box.
 * @return Area of intersection in range from 0 to 1.
 * **/
static float iou(const cv::Rect2f &bboxFiltered, const cv::Rect2f &bboxPredicted) {
    auto in = (bboxFiltered & bboxPredicted).area();  //  Find rectangles area intersection.
    auto un = bboxFiltered.area() + bboxPredicted.area() - in;
    if (un < DBL_EPSILON) {
        return 0;
    }
    return in / un;
}

/** @brief Calculate the tensor of intersection distances between predicted and filtered bboxes.
 * @param predictedBoxes List of predicted bounding boxes.
 * @param filteredBoxes List of bounding boxes after Kalman filtering.
 * @return Tensor(matrix) of intersection distances.
 * **/
static torch::Tensor iou_distance(const std::vector<cv::Rect2f> &predictedBoxes, const std::vector<cv::Rect2f> &filteredBoxes) {
    size_t tracks_n = filteredBoxes.size();                              //> Size of filtered bboxes
    size_t pred_n = predictedBoxes.size();                               //> Size of predicted bboxes
    auto distance = torch::empty({int64_t(tracks_n), int64_t(pred_n)});  //> Create matrix of iou distance
    for (size_t i = 0; i < tracks_n; i++) {
        for (size_t j = 0; j < pred_n; j++) {
            //> Area of not intersected rects in range from 0 to 1.
            distance[i][j] = 1 - alpr::iou(filteredBoxes[i], predictedBoxes[j]);
        }
    }
    return distance;
}

}  // namespace alpr