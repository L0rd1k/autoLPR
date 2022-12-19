#pragma once 

#include <string>

#include "opencv2/opencv.hpp"

#include "handlers/handler.h"
#include "utils/timer.h"

namespace alpr {

class VideoGrabber {
public:
    VideoGrabber() = default;
    VideoGrabber(const std::string path);

private:
    void readLoop();
    alpr::Handler handler_;
    std::string streamPath_;
    cv::VideoCapture cap_;
};

}