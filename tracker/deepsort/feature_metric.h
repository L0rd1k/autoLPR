#pragma once

#include <stdint.h>
#include <torch/torch.h>

#include <vector>

namespace alpr {

static torch::Tensor cos_distance(torch::Tensor x, torch::Tensor y) {
    return std::get<0>(torch::min(1 - torch::matmul(x, y.t()), 0)).cpu();
}

template <typename TrackData>
class FeatureMetric {
public:
    FeatureMetric(std::vector<TrackData> &data)
        : _data(data) {
        std::cout << ">   Cnctr: Feature metric" << std::endl;
    }
    /** @brief 
     * @param features 
     * @param targets Id of trackers boxes. 
     * @return Tensor of distances. **/
    torch::Tensor distance(torch::Tensor features, const std::vector<int> &targets) {
        auto distance = torch::empty({int64_t(targets.size()), features.size(0)});
        if (features.size(0)) {
            for (size_t i = 0; i < targets.size(); ++i) {
                distance[i] = alpr::cos_distance(_data[targets[i]].feature.get(), features);
            }
        }
        return distance;
    }

    void update(torch::Tensor feature, const std::vector<int> &targets) {
        for (size_t i = 0; i < targets.size(); ++i) {
            _data[targets[i]].feature.add(feature[i]);
        }
    }

private:
    std::vector<TrackData> &_data;
};

}  // namespace alpr