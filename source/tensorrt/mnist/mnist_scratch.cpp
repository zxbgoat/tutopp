//
// Created by tesla on 2021/6/6.
//

#include "mnist.h"


ScratchMNIST::ScratchMNIST(const MNISTParams &params)
{
    this->params = params;
    this->number = 0;
    this->engine = nullptr;
}


bool ScratchMNIST::build()
{
    weights = loadweights(params.weightfile);
    auto builder = uniptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) return false;
    auto network = uniptr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network) return false;
    auto config = uniptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // 1. create input tensor of shape { 1, 1, 28, 28 }
    string inputname = params.innames[0];
    Dims3 dimensions{1, params.inh, params.inw};
    ITensor* data = network->addInput(inputname.c_str(), DataType::kFLOAT, dimensions);
    // 2. create scale layer with default power/shift and specified scale parameter
    const float scaleparam = 0.0125f;
    const Weights power{DataType::kFLOAT, nullptr, 0};
    const Weights shift{DataType::kFLOAT, nullptr, 0};
    const Weights scale{DataType::kFLOAT, &scaleparam, 1};
    IScaleLayer* scale1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
    // 3. add convolution layer with 20 outputs and a 5x5 filter
    Dims kernelsize{2, {5, 5}, {}};
    auto kernelweights = weights["conv1filter"];
    auto biasweights = weights["conv1bias"];
    IConvolutionLayer* conv1 = network->addConvolutionNd(*scale1->getOutput(0), 20, kernelsize, kernelweights, biasweights);
    conv1->setStride(DimsHW{1, 1});
    // 4. add max pooling layer with stride of 2x2 and kernel size of 2x2.
    Dims pollsize{2, {2, 2}, {}};
    IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kMAX, pollsize);
    pool1->setStride(DimsHW{2, 2});
    // Add second convolution layer with 50 outputs and a 5x5 filter.
    IConvolutionLayer* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 50, Dims{2, {5, 5}, {}}, weights["conv2filter"], weights["conv2bias"]);
    conv2->setStride(DimsHW{1, 1});
    // Add second max pooling layer with stride of 2x2 and kernel size of 2x3>
    IPoolingLayer* pool2 = network->addPoolingNd(*conv2->getOutput(0), PoolingType::kMAX, Dims{2, {2, 2}, {}});
    pool2->setStride(DimsHW{2, 2});
    // Add fully connected layer with 500 outputs.
    IFullyConnectedLayer* ip1 = network->addFullyConnected(*pool2->getOutput(0), 500, weights["ip1filter"], weights["ip1bias"]);
    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
    // Add second fully connected layer with 20 outputs.
    IFullyConnectedLayer* ip2 = network->addFullyConnected(*relu1->getOutput(0), params.outsize, weights["ip2filter"], weights["ip2bias"]);
    // Add softmax layer to determine the probability.
    ISoftMaxLayer* prob = network->addSoftMax(*ip2->getOutput(0));
    prob->getOutput(0)->setName(params.outnames[0].c_str());
    network->markOutput(*prob->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(params.batchsize);
    config->setMaxWorkspaceSize(16_MiB);
    if (params.fp16) config->setFlag(BuilderFlag::kFP16);
    if (params.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 64.0f, 64.0f);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), params.dlacore);
    engine = shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!engine) return false;
    assert(network->getNbInputs() == 1);
    auto inputDims = network->getInput(0)->getDimensions();
    assert(inputDims.nbDims == 3);
    assert(network->getNbOutputs() == 1);
    auto outputDims = network->getOutput(0)->getDimensions();
    assert(outputDims.nbDims == 3);
    return true;
}


wtsmap ScratchMNIST::loadweights(const string& filepath)
{
    gLogInfo << "Loading weights from " << filepath << endl;
    ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) throw "Unable to open the weights file";
    int32_t count;
    in >> count;  // get the number of weight blobs
    if (count <= 0) throw "Invalid weight map file";
    wtsmap weights;
    while (count--)
    {
        nvinfer1::Weights wt{DataType::kFLOAT, nullptr, 0};
        int type;
        uint32_t size;
        string name;
        in >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT) {}
        if (wt.type == DataType::kHALF) {}
        wt.count = size;
        weights[name] = wt;
    }
    return weights;
}


bool ScratchMNIST::preprocess(const BufferManager &buffers)
{
}