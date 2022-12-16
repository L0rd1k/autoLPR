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
    alpr::Timer timer;
    while(true) {
        cv::Mat img;
        cap_.read(img);
        if(!img.empty()) {
            detector_.process(img);
            if(timer.duration() > 1) {
                std::cout << i << std::endl;
                i = 0;
                timer.reset();
            } 
            i++;
        } 

    }
}

