#pragma once

#include <stdint.h>
#include <torch/torch.h>

namespace alpr {

class FeatureBundle {
public:
    FeatureBundle()
        : _isFull(false),
          _next(0),
          _store(torch::empty({budget, feature_sz}).cpu()) {
    }

    void clear() {
        _next = 0;
        _isFull = false;
    }

    bool empty() const {
        return _next == 0 && !_isFull;
    }

    void add(torch::Tensor feature) {
        if (_next == budget) {
            _isFull = true;
            _next = 0;
        }
        _store[_next++] = feature;
    }

    torch::Tensor get() const {
        if (_isFull) {
            return _store;
        }
        return _store.slice(0, 0, _next);
    }

private:
    static const int64_t budget = 100;
    static const int64_t feature_sz = 512;
    bool _isFull;
    int64_t _next;
    torch::Tensor _store;
};

}  // namespace alpr