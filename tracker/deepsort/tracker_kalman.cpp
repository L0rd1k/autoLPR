#include "tracker_kalman.h"

int alpr::TrackerKalman::count = 0;

alpr::TrackerKalman::TrackerKalman() {
    int state_n = 7;
    int measure_n = 4;

    kalman_ = cv::KalmanFilter(state_n, measure_n, 0);
    measurement_ = cv::Mat::zeros(measure_n, 1, CV_32F);
    kalman_.transitionMatrix = (cv::Mat_<float>(state_n, state_n) << 1, 0, 0, 0, 1, 0, 0,
                                0, 1, 0, 0, 0, 1, 0,
                                0, 0, 1, 0, 0, 0, 1,
                                0, 0, 0, 1, 0, 0, 0,
                                0, 0, 0, 0, 1, 0, 0,
                                0, 0, 0, 0, 0, 1, 0,
                                0, 0, 0, 0, 0, 0, 1);

    setIdentity(kalman_.measurementMatrix);
    setIdentity(kalman_.processNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(kalman_.errorCovPost, cv::Scalar::all(1));
}

alpr::TrackerKalman::TrackerKalman(cv::Rect2f box) {
    init(box);
}

void alpr::TrackerKalman::init(cv::Rect2f box) {
    kalman_.statePost.at<float>(0, 0) = box.x + box.width / 2;
    kalman_.statePost.at<float>(1, 0) = box.y + box.height / 2;
    kalman_.statePost.at<float>(2, 0) = box.area();
    kalman_.statePost.at<float>(3, 0) = box.width / box.height;
}

void alpr::TrackerKalman::predict() {
    ++update_timer;
    kalman_.predict();
}

void alpr::TrackerKalman::update(cv::Rect2f state_box) {
    update_timer = 0;
    ++hits;
    if (k_state_ == alpr::KalmanState::Unconfirmed && hits > init_counts) {
        k_state_ = alpr::KalmanState::Confirmed;
        id_ = count++;
    }
    measurement_.at<float>(0, 0) = state_box.x + state_box.width / 2;
    measurement_.at<float>(1, 0) = state_box.y + state_box.height / 2;
    measurement_.at<float>(2, 0) = state_box.area();
    measurement_.at<float>(3, 0) = state_box.width / state_box.height;
    kalman_.correct(measurement_);
}

void alpr::TrackerKalman::miss() {
    if (k_state_ == alpr::KalmanState::Unconfirmed) {
        k_state_ == alpr::KalmanState::Removed;
    } else if (update_timer > max_stage) {
        k_state_ == alpr::KalmanState::Removed;
    }
}

cv::Rect2f alpr::TrackerKalman::getBox() const {
    auto cx = kalman_.statePost.at<float>(0, 0), cy = kalman_.statePost.at<float>(1, 0), s = kalman_.statePost.at<float>(2, 0), r = kalman_.statePost.at<float>(3, 0);
    float w = sqrt(s * r);
    float h = s / w;
    float x = (cx - w / 2);
    float y = (cy - h / 2);
    return cv::Rect2f(x, y, w, h);
}

int alpr::TrackerKalman::getId() const {
    return id_;
}