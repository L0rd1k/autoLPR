#include "yolo_detector.h"

alpr::YoloDetector::YoloDetector() {
    std::string yoloModelPath = "/home/ilya/yolov3_tiny.cfg";
    std::string yoloWeightsPathTV = "/home/ilya/yolov3-tiny.weights";
    yoloDetector_ = create(yoloModelPath, yoloWeightsPathTV);
}

alpr::PredictionStatus alpr::YoloDetector::detect(const cv::Mat& frame) {
    std::vector<alpr::PredictedBox> predictedBoxes;
    alpr::PredictionStatus prediction;
    if (yoloDetector_->predict(frame, predictedBoxes)) {
        if (predictedBoxes.size()) {
            for (auto& box : predictedBoxes) {
                auto rect = box.getROI();
                prediction.rects.push_back(rect);
            }
            prediction.frame = frame;
            prediction.found = true;
        }
    }
    return prediction;
}

std::shared_ptr<alpr::CPUYoloDetector> alpr::YoloDetector::create(const std::string yoloModelPath, const std::string yoloWeightsPathTV) {
    std::shared_ptr<alpr::CPUYoloDetector> detector = std::make_shared<alpr::CPUYoloDetector>(yoloModelPath, yoloWeightsPathTV);
    if (!detector->init()) {
        std::cerr << "Couldn't initialize network" << std::endl;
        return nullptr;
    }
    return detector;
}
