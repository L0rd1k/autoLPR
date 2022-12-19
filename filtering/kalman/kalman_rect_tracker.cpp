#include "kalman_rect_tracker.h"

alpr::KalmanRectTracker::KalmanRectTracker(int dynamicParams, int measureParams, int controlParams)
    : dynamicParams_(dynamicParams),
      measureParams_(measureParams),
      controlParams_(controlParams) {
}

alpr::KalmanRectTracker::~KalmanRectTracker() {
}

void alpr::KalmanRectTracker::init() {
    kalman_f.init(dynamicParams_, measureParams_, controlParams_);
    dynamicMat_ = cv::Mat(dynamicParams_, 1, CV_32F);
    measureMat_ = cv::Mat(measureParams_, 1, CV_32F);
    noiseMat_ = cv::Mat(dynamicParams_, 1, CV_32F);

    cv::setIdentity(kalman_f.transitionMatrix);

    kalman_f.measurementMatrix = (cv::Mat_<float>(measureParams_, dynamicParams_) << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    kalman_f.processNoiseCov = (cv::Mat_<float>(dynamicParams_, dynamicParams_) << 0.01, 0, 0, 0, 0, 0,
                                0, 0.01, 0, 0, 0, 0,
                                0, 0, 5.0f, 0, 0, 0,
                                0, 0, 0, 5.0f, 0, 0,
                                0, 0, 0, 0, 0.01, 0,
                                0, 0, 0, 0, 0, 0.01);

    cv::setIdentity(kalman_f.measurementNoiseCov, cv::Scalar(1e-1));
}

cv::Rect alpr::KalmanRectTracker::update(cv::Rect rect) {
    float dt = _timer.duration();
    _timer.reset();

    kalman_f.transitionMatrix.at<float>(2) = dt;
    kalman_f.transitionMatrix.at<float>(9) = dt;

    measureMat_.at<float>(0) = rect.x + rect.width / 2;
    measureMat_.at<float>(1) = rect.y + rect.height / 2;
    measureMat_.at<float>(2) = (float)rect.width;
    measureMat_.at<float>(3) = (float)rect.height;

    if (!is_inited()) {
        kalman_f.errorCovPre = (cv::Mat_<float>(dynamicParams_, dynamicParams_) << 1.f, 0, 0, 0, 0, 0,
                                0, 1.f, 0, 0, 0, 0,
                                0, 0, 1.f, 0, 0, 0,
                                0, 0, 0, 1.f, 0, 0,
                                0, 0, 0, 0, 1.f, 0,
                                0, 0, 0, 0, 0, 1.f);

        dynamicMat_.at<float>(0) = measureMat_.at<float>(0);
        dynamicMat_.at<float>(1) = measureMat_.at<float>(1);
        dynamicMat_.at<float>(2) = 0;
        dynamicMat_.at<float>(3) = 0;
        dynamicMat_.at<float>(4) = measureMat_.at<float>(2);
        dynamicMat_.at<float>(5) = measureMat_.at<float>(3);
        kalman_f.statePre = dynamicMat_;
        kalman_f.statePost = dynamicMat_;
        inited_ = true;
    }

    kalman_f.correct(measureMat_);
    dynamicMat_ = kalman_f.predict();

    return cv::Rect(dynamicMat_.at<float>(0) - (dynamicMat_.at<float>(4) * 0.5),
                    dynamicMat_.at<float>(1) - (dynamicMat_.at<float>(5) * 0.5),
                    dynamicMat_.at<float>(4),
                    dynamicMat_.at<float>(5));
}

bool alpr::KalmanRectTracker::is_inited() {
    return inited_;
}