#include "sort.h"

alpr::Sort::Sort(const std::array<int64_t, 2> &sz)
    : manager_(std::make_unique<Manager>(data_, sz)) {
}

alpr::Sort::~Sort() {
}

std::vector<alpr::TrackerState> alpr::Sort::update(const std::vector<cv::Rect2f> &pred) {
}