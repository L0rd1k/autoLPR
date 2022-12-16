#include "morphology_detector.h"

alpr::MorphologyDetector::MorphologyDetector() {
    _detector = std::make_shared<alpr::Detector>();
    _tracker = std::make_shared<alpr::Tracker>();
    _tracker->create(alpr::TrackerType::MEDIANFLOW);
}

bool compareContourAreas(std::vector<cv::Point>& contour1, std::vector<cv::Point>& contour2) {
    const double i = fabs(contourArea(cv::Mat(contour1)));
    const double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}

bool alpr::MorphologyDetector::process(cv::Mat& img) {
    if (_detectorResult == nullptr) {
        _detectorResult = std::make_shared<std::future<alpr::PredictionStatus>>(
            std::async(std::launch::async, &alpr::Detector::detect, _detector, img));
    }

    alpr::PredictionStatus prediction;
    if (_detectorResult->valid() == true) {
        auto state = _detectorResult->wait_for(std::chrono::milliseconds(0));
        if (state == std::future_status::ready) {
            prediction = _detectorResult->get();
            if (!prediction.rects.empty()) {
                for (auto& boxs : prediction.rects) {
                    cv::rectangle(img, boxs, cv::Scalar(0, 0, 255), 3);
                }
            }
            _detectorResult.reset();
        }
    }

    cv::imshow("Result", img);
    cv::waitKey(1);
    return true;

    // cv::Mat dst;
    // img.convertTo(dst, -1, 2, 0);

    // cv::Mat greyFrame, morphFrame;
    // cv::cvtColor(img, greyFrame, CV_BGR2GRAY);

    // cv::Mat rectangleKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5));
    // cv::morphologyEx(greyFrame, morphFrame, cv::MORPH_BLACKHAT, rectangleKernel);
    // // cv::imshow("morphFrame", morphFrame);

    // cv::Mat lightFrame;
    // cv::Mat squareKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 10));
    // cv::morphologyEx(morphFrame, lightFrame, cv::MORPH_CLOSE, squareKernel);
    // cv::threshold(lightFrame, lightFrame, 0, 255, cv::THRESH_OTSU);
    // // cv::imshow("lightFrame", lightFrame);

    // cv::Mat gradX;
    // double minVal, maxVal;
    // int dx = 1, dy = 0, ddepth = CV_32F, ksize = -1;
    // cv::Sobel(morphFrame, gradX, ddepth, dx, dy, ksize);  // Looks coarse if imshow, because the range is high?
    // gradX = cv::abs(gradX);
    // // cv::imshow("Sobel1", gradX);

    // cv::minMaxLoc(gradX, &minVal, &maxVal);
    // gradX = 255 * ((gradX - minVal) / (maxVal - minVal));
    // gradX.convertTo(gradX, CV_8U);

    // // // cv::GaussianBlur(gradX, gradX, cv::Size(5, 5), 0);

    // cv::morphologyEx(gradX, gradX, cv::MORPH_CLOSE, rectangleKernel);
    // cv::threshold(gradX, gradX, 0, 255, cv::THRESH_OTSU);

    // // // cv::erode(gradX, gradX, 2);
    // cv::dilate(gradX, gradX, 2);

    // cv::bitwise_and(gradX, gradX, lightFrame);
    // cv::dilate(gradX, gradX, 2);
    // cv::erode(gradX, gradX, 1);

    // std::vector<std::vector<cv::Point>> contours;
    // cv::findContours(gradX, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // std::sort(contours.begin(), contours.end(), compareContourAreas);
    // std::vector<std::vector<cv::Point>> top_contours;
    // top_contours.assign(contours.end() - 50, contours.end()); // Descending order

    // std::cout << contours.size() << std::endl;
    // std::vector<cv::Rect> rectangles;
    // for (std::vector<cv::Point> currentCandidate : top_contours) {
    //     cv::Rect temp = cv::boundingRect(currentCandidate);
    //     float difference = temp.area() - cv::contourArea(currentCandidate);
    //     if (difference < 4000) {
    //         rectangles.push_back(temp);
    //     }
    // }

    // rectangles.erase(std::remove_if(rectangles.begin(), rectangles.end(), [](cv::Rect temp) {
    //     const float aspect_ratio = temp.width / (float) temp.height;
    //     return aspect_ratio < 1 || aspect_ratio > 10;
    // }), rectangles.end());

    // for(auto r : rectangles) {
    //     cv::rectangle(img, r, cv::Scalar(0, 255, 0));
    // }
}
