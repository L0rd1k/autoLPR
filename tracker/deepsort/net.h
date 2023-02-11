#pragma once

#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <string>

namespace alpr {

void loadSequential(torch::nn::Sequential s, std::ifstream &fs);
void loadConv2d(torch::nn::Conv2d m, std::ifstream &fs);
void loadTensor(torch::Tensor t, std::ifstream &fs);
void loadBatchNorm(torch::nn::BatchNorm2d m, std::ifstream &fs);
torch::nn::Sequential makeLayers(int64_t cIn, int64_t cOut, size_t repeatTimes, bool isDownsample = false);

class BasicBlockImpl : public torch::nn::Module {
public:
    BasicBlockImpl(int64_t cIn, int64_t cOut, bool isDownsample = false);
    torch::Tensor forward(torch::Tensor tnsr);
    torch::nn::Sequential conv{nullptr}, downsample{nullptr};
};

TORCH_MODULE(BasicBlock);

class NetImpl : public torch::nn::Module {
public:
    NetImpl();
    torch::Tensor forward(torch::Tensor tnsr);
    void loadForm(const std::string &binPath);

private:
    void createLayers();
    torch::nn::Sequential conv1{nullptr}, conv2{nullptr};
};

TORCH_MODULE(Net);

}  // namespace alpr
