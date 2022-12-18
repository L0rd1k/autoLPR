
#include "cpu_yolo_detector.h"

alpr::CPUYoloDetector::CPUYoloDetector(std::string cfgFile, std::string weightsFile)
    : cfgFile_(cfgFile),
      weightsFile_(weightsFile) {
}

alpr::CPUYoloDetector::~CPUYoloDetector() {
}

void alpr::CPUYoloDetector::setModelCfgPath(const std::string file) {
    cfgFile_ = file;
}

const std::string alpr::CPUYoloDetector::getModelCfgPath() {
    return cfgFile_;
}

void alpr::CPUYoloDetector::setWeightPath(const std::string file) {
    weightsFile_ = file;
}

const std::string alpr::CPUYoloDetector::setWeightPath() {
    return weightsFile_;
}

bool alpr::CPUYoloDetector::isExist(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool alpr::CPUYoloDetector::init() {
    if (isExist(cfgFile_) && isExist(weightsFile_)) {
        net_ = cv::dnn::readNetFromDarknet(cfgFile_, weightsFile_);
    }
    if (net_.empty()) {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "cfg-file:     " << cfgFile_ << std::endl;
        std::cerr << "weights-file: " << weightsFile_ << std::endl;
        exit(-1);
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    outLayerNames_ = getOutputsNames(net_);
    return true;
}

bool alpr::CPUYoloDetector::predict(const cv::Mat& img, std::vector<alpr::PredictedBox>& predictions) {
    if (img.empty()) {
        return false;
    }
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1 / 255.0, cvSize(inpWidth_, inpHeight_), cv::Scalar(0, 0, 0), true, false);
    net_.setInput(blob);
    std::vector<cv::Mat> outs;
    net_.forward(outs, outLayerNames_);
    predictions = postprocess(outs, img.size());
    for(auto &elem : predictions) {
        elem.setROI(fixBound(elem.getROI(), img.size()));
    }
    return true;
}

cv::Rect alpr::CPUYoloDetector::fixBound(cv::Rect r, cv::Size imageSize) {
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

std::vector<cv::String> alpr::CPUYoloDetector::getOutputsNames(const cv::dnn::Net& net) {
    static std::vector<cv::String> names;
    if (names.empty()) {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<cv::String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}

std::vector<alpr::PredictedBox> alpr::CPUYoloDetector::postprocess(const std::vector<cv::Mat>& outs, const cv::Size size) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold_) {
                int centerX = (int)(data[0] * size.width);
                int centerY = (int)(data[1] * size.height);
                int width = (int)(data[2] * size.width);
                int height = (int)(data[3] * size.height);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                if(classIdPoint.x == 2) {
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }
    std::vector<alpr::PredictedBox> predictions;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indices);
    for (size_t idx = 0; idx < indices.size(); ++idx) {
        alpr::PredictedBox prediction(classIds[idx], confidences[idx], boxes[idx]);
        predictions.push_back(prediction);
    }
    return predictions;
}
