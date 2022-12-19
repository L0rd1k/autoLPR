#pragma once

#include "opencv2/opencv.hpp"

namespace alpr {

class Recognizer {
public:
    Recognizer();
    bool process(cv::Mat& img, cv::Rect2d rect);
    static bool compareContourAreas(std::vector<cv::Point>& contour1, std::vector<cv::Point>& contour2);
};

}  // namespace alpr