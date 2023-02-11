#pragma once

#include <memory>
#include <mutex>

namespace ds {

template <typename T>
class RingQueue {
public:
    explicit RingQueue(uint64_t size)
        : buffer_(std::unique_ptr<T[]>(new T[size])),
          maxQueueSize_(size) {
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);
        head_ = tail_;
        isFull_ = false;
    }

    void push(T& item) {
        std::lock_guard<std::mutex> lock(mtx_);
        buffer_[head_] = item;
        if (isFull_) {
            tail_ = (tail_ + 1) % maxQueueSize_;
        }
        head_ = (head_ + 1) % maxQueueSize_;
        isFull_ = head_ == tail_;
    }

    T get() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (isEmpty()) {
            return T();
        }
        auto value = buffer_[tail_];
        isFull_ = false;
        tail_ = (tail_ + 1) % maxQueueSize_;
        return value;
    }

    bool isEmpty() {
        return (!isFull_ && (head_ == tail_));
    }

    bool isFull() const {
        return isFull_;
    }

    uint64_t capacity() const {
        return maxQueueSize_;
    }

    uint64_t size() const {
        uint64_t size = maxQueueSize_;
        if (!isFull_) {
            if (head_ >= tail_) {
                size = head_ - tail_;
            } else {
                size = maxQueueSize_ + head_ - tail_;
            }
        }
        return size;
    }

private:
    std::unique_ptr<T[]> buffer_;
    uint64_t maxQueueSize_;
    uint64_t head_ = 0;
    uint64_t tail_ = 0;
    bool isFull_ = false;
    std::mutex mtx_;
};

}  // namespace ds
