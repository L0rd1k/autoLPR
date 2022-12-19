#include "video_grabber.h"

alpr::VideoGrabber::VideoGrabber(const std::string path)
    : streamPath_(path) {
    cap_.open(streamPath_);
    if (!cap_.isOpened()) {
        std::cerr << "Can't open video stream" << std::endl;
    } else {
        readLoop();
    }
}

void alpr::VideoGrabber::readLoop() {
    int i = 0;
    while(true) {
        cv::Mat img;
        cap_.read(img);
        if(!img.empty()) {
            handler_.process(img);
        } 

    }
}

