#pragma once

#include <chrono>

namespace alpr {

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    void start() {
        start_ = std::chrono::steady_clock::now();
    }
    void reset() {
        start_ = std::chrono::steady_clock::now();
    }

    double duration() const {
        return std::chrono::duration_cast<sec>(std::chrono::steady_clock::now() - start_).count();
    }
    double durationMs() const {
        return duration() * 1000;
    }

private:
    using sec = std::chrono::duration<double, std::ratio<1>>;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

}  // namespace alpr