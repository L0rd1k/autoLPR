#include "net.h"

alpr::BasicBlockImpl::BasicBlockImpl(int64_t cIn, int64_t cOut, bool isDownsample) {
    conv = register_module(
        "conv", torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(cIn, cOut, 3).stride(isDownsample ? 2 : 1).padding(1).bias(false)),
                    torch::nn::BatchNorm2d(cOut), torch::nn::Functional(torch::relu),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(cOut, cOut, 3).stride(1).padding(1).bias(false)),
                    torch::nn::BatchNorm2d(cOut)));
    if (isDownsample) {
        downsample = register_module("downsample",
                                     torch::nn::Sequential(
                                         torch::nn::Conv2d(torch::nn::Conv2dOptions(cIn, cOut, 1).stride(2).bias(false)),
                                         torch::nn::BatchNorm2d(cOut)));
    } else if (cIn != cOut) {
        downsample = register_module("downsample",
                                     torch::nn::Sequential(
                                         torch::nn::Conv2d(torch::nn::Conv2dOptions(cIn, cOut, 1).stride(1).bias(false)),
                                         torch::nn::BatchNorm2d(cOut)));
    }
}

torch::Tensor alpr::BasicBlockImpl::forward(torch::Tensor tnsr) {
    auto y = conv->forward(tnsr);
    if (!downsample.is_empty()) {
        tnsr = downsample->forward(tnsr);
    }
    return torch::relu(tnsr + y);
}

void alpr::loadTensor(torch::Tensor t, std::ifstream &fs) {
    fs.read(static_cast<char *>(t.data_ptr()), t.numel() * sizeof(float));
}

void alpr::loadConv2d(torch::nn::Conv2d m, std::ifstream &fs) {
    loadTensor(m->weight, fs);
    if (m->options.bias()) {
        loadTensor(m->bias, fs);
    }
}

void alpr::loadBatchNorm(torch::nn::BatchNorm2d m, std::ifstream &fs) {
    loadTensor(m->weight, fs);
    loadTensor(m->bias, fs);
    loadTensor(m->running_mean, fs);
    loadTensor(m->running_var, fs);
}

void alpr::loadSequential(torch::nn::Sequential s, std::ifstream &fs) {
    if (s.is_empty()) {
        return;
    }
    for (auto &m : s->children()) {
        if (auto c = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m)) {
            loadConv2d(c, fs);
        } else if (auto b = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m)) {
            loadBatchNorm(b, fs);
        }
    }
}

torch::nn::Sequential alpr::makeLayers(int64_t cIn, int64_t cOut, size_t repeatTimes, bool isDownsample) {
    torch::nn::Sequential result;
    for (size_t i = 0; i < repeatTimes; ++i) {
        result->push_back(alpr::BasicBlock(i == 0 ? cIn : cOut, cOut, i == 0 ? isDownsample : false));
    }
    return result;
}

alpr::NetImpl::NetImpl() {
    std::cout << ">   Cnctr: Net implementation" << std::endl;
    createLayers();
}

void alpr::NetImpl::createLayers() {
    conv1 = register_module("conv1", torch::nn::Sequential(
                                         torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1)),
                                         torch::nn::BatchNorm2d(64),
                                         torch::nn::Functional(torch::relu)));
    conv2 = register_module("conv2", torch::nn::Sequential());
    conv2->extend(*makeLayers(64, 64, 2, false));
    conv2->extend(*makeLayers(64, 128, 2, true));
    conv2->extend(*makeLayers(128, 256, 2, true));
    conv2->extend(*makeLayers(256, 512, 2, true));
}

torch::Tensor alpr::NetImpl::forward(torch::Tensor tnsr) {
    tnsr = conv1->forward(tnsr);
    tnsr = torch::max_pool2d(tnsr, 3, 2, 1);
    tnsr = conv2->forward(tnsr);
    tnsr = torch::avg_pool2d(tnsr, {8, 4}, 1);
    tnsr = tnsr.view({tnsr.size(0), -1});
    tnsr.div_(tnsr.norm(2, 1, true));
    return tnsr;
}

void alpr::NetImpl::loadForm(const std::string &binPath) {
    std::ifstream fs(binPath, std::ios_base::binary);
    loadSequential(conv1, fs);
    for(auto &m : conv2->children()) {
        auto b = std::static_pointer_cast<BasicBlockImpl>(m);
        loadSequential(b->conv, fs);
        loadSequential(b->downsample, fs);
    }
    fs.close();
}
