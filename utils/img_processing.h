#pragma once

#include "opencv2/opencv.hpp"

static cv::Rect fixBound(cv::Rect r, cv::Size imageSize) {
    r.x = cv::max(r.x, 0);
    r.y = cv::max(r.y, 0);
    r.x = cv::min(r.x, imageSize.width);
    r.y = cv::min(r.y, imageSize.height);
    if ((r.x + r.width) > imageSize.width) {
        r.width = r.width - ((r.x + r.width) - imageSize.width);
    }
    if ((r.y + r.height) > imageSize.height) {
        r.height = r.height - ((r.y + r.height) - imageSize.height);
    }
    return r;
}
