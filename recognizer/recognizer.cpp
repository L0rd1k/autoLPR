#include "recognizer.h"

alpr::Recognizer::Recognizer() {
}

bool alpr::Recognizer::compareContourAreas(std::vector<cv::Point>& contour1, std::vector<cv::Point>& contour2) {
    const double i = fabs(contourArea(cv::Mat(contour1)));
    const double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}

bool alpr::Recognizer::process(cv::Mat& img, cv::Rect2d rect) {
    cv::Mat dstImg, carImg = img(rect);
    carImg.convertTo(dstImg, -1, 2, 0);

    cv::Mat greyFrame, morphFrame;
    cv::cvtColor(carImg, greyFrame, CV_BGR2GRAY);
    cv::Mat rectangleKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5));
    cv::morphologyEx(greyFrame, morphFrame, cv::MORPH_BLACKHAT, rectangleKernel);
    cv::Mat lightFrame;
    cv::Mat squareKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 10));
    cv::morphologyEx(morphFrame, lightFrame, cv::MORPH_CLOSE, squareKernel);
    cv::threshold(lightFrame, lightFrame, 0, 255, cv::THRESH_OTSU);
    cv::Mat gradX;
    double minVal, maxVal;
    int dx = 1, dy = 0, ddepth = CV_32F, ksize = -1;
    cv::Sobel(morphFrame, gradX, ddepth, dx, dy, ksize);  // Looks coarse if imshow, because the range is high?
    gradX = cv::abs(gradX);
    cv::minMaxLoc(gradX, &minVal, &maxVal);
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal));
    gradX.convertTo(gradX, CV_8U);
    cv::morphologyEx(gradX, gradX, cv::MORPH_CLOSE, rectangleKernel);
    cv::threshold(gradX, gradX, 0, 255, cv::THRESH_OTSU);
    cv::imshow("Grad X", gradX);
}
