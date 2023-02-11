#include <iostream>

#include "grabber/video_grabber.h"
#include "utils/size.h"

#include <torch/torch.h>

#include "utils/ring_queue.h"

int main(int argc, char **argv) {
    alpr::VideoGrabber grabber("/home/ilya/Загрузки/Telegram Desktop/Mix_x264.avi");
    return 0;
}
