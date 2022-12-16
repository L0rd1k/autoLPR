#pragma once

#include <opencv2/core.hpp>

namespace alpr {

struct PredictedBox {
    PredictedBox() = default;
    PredictedBox(uint8_t id, double confidence, cv::Rect roi)
        : id_(id),
          confidence_(confidence),
          roi_(roi) {
    }
    uint8_t getClassId() const {
        return id_;
    }

    void setClassId(uint8_t id) {
        id_ = id;
    }

    double getConfidence() const {
        return confidence_;
    }

    void setConfidence(double confidence) {
        confidence_ = confidence;
    }

    cv::Rect getROI() const {
        return roi_;
    }

    void setROI(cv::Rect roi) {
        roi_ = roi;
    }

private:
    uint8_t id_ = 0;
    double confidence_ = 0;
    cv::Rect roi_;
};

}  // namespace alpr
