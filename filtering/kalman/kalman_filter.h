#pragma once 

#include <opencv2/opencv.hpp>
#include "utils/timer.h"

namespace alpr {

class KalmanFilter {
public:
    KalmanFilter();
    virtual ~KalmanFilter();
    virtual void init() = 0;
    virtual cv::Rect update(cv::Rect rect) = 0;
protected:
    cv::KalmanFilter kalman_f;
    cv::Mat dynamicMat_;
    cv::Mat measureMat_;
    alpr::Timer _timer;
};

}