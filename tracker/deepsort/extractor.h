#pragma once

#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <vector>

#include "net.h"

namespace alpr {

class Extractor {
public:
    Extractor(std::string weights) {
        std::cout << ">   Cnctr: Extractor" << std::endl;
        net->loadForm(weights);
        // net->to(torch::kCUDA);
        net->to(torch::kCPU);
        net->eval();
    }

    /** @brief
     * @param input Template object image.
     * @return  **/
    torch::Tensor extract(std::vector<cv::Mat> input) {
        if (input.empty()) {
            return torch::empty({0, 512});
        }
        static const torch::Tensor mean = torch::tensor({0.485f, 0.456f, 0.406f}).view({1, -1, 1, 1}).cpu();
        static const torch::Tensor std = torch::tensor({0.229f, 0.224f, 0.225f}).view({1, -1, 1, 1}).cpu();
        std::vector<torch::Tensor> resized;  //> Tenosrs vector of images to track
        for (auto &x : input) {
            cv::resize(x, x, {64, 128});  //> Resize image to size 64x128
            cv::cvtColor(x, x, cv::COLOR_RGB2BGR);
            x.convertTo(x, CV_32F, 1.0 / 255);
            resized.push_back(torch::from_blob(x.data, {128, 64, 3}));
        }
        std::cout << "  > Resized images: " << resized.size() << std::endl;
        auto tensor = torch::stack(resized).cpu().permute({0, 3, 1, 2}).sub_(mean).div_(std);
        return net(tensor);
    }
private:
    alpr::Net net;
};

}  // namespace alpr