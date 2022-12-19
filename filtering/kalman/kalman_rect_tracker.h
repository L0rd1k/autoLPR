#pragma once

#include "kalman_filter.h"

namespace alpr {

class KalmanRectTracker : public KalmanFilter {
public:
    KalmanRectTracker() = default;
    KalmanRectTracker(int dynamicParams = 6, int measureParams = 4, int controlParams = 0);
    virtual ~KalmanRectTracker();
    void init() override;
    cv::Rect update(cv::Rect rect) override;
    bool is_inited();
private:
    cv::Mat dynamicMat_;
    cv::Mat measureMat_;
    cv::Mat noiseMat_;
    int dynamicParams_;
    int measureParams_;
    int controlParams_;
    bool inited_;
};

}  // namespace alpr