//
// Created by tesla on 2021/6/6.
//

#ifndef TUTO_MNIST_H
#define TUTO_MNIST_H

#include "comm.h"

struct MNISTParams
{
    int inh, inw;
    int outsize;
    bool fp16, int8;
    string weightfile;
    string meansproto;
    vector<string> innames;
    vector<string> outnames;
    int batchsize, dlacore;
    string datadir;
};


class ScratchMNIST
{
public:
    explicit ScratchMNIST(const MNISTParams& params);
    bool build();
    bool infer();

protected:
    static wtsmap loadweights(const string& filepath);
    bool preprocess(const BufferManager& buffers);
    bool postprocess(const BufferManager& buffers);

private:
    MNISTParams params;
    int number;
    wtsmap weights;
    shared_ptr<nvinfer1::ICudaEngine> engine;

};


class PluginMNIST
{
public:
    PluginMNIST(const MNISTParams& params);
    bool build();
    bool infer();

protected:
    bool preprocess();
    bool postprocess();

private:
    shared_ptr<nvinfer1::ICudaEngine> engine;
    MNISTParams params;
    uniptr<nvcaffeparser1::IBinaryProtoBlob> meanblob;
    nvinfer1::Dims indims;
};


class DynamicReshape
{
public:
    DynamicReshape(MNISTParams  params): params(std::move(params)) {}

    bool build();
    bool prepare();
    bool infer();

private:
    bool buildPreprocessorEngine(const uniptr<nvinfer1::IBuilder>& builder);
    bool buildPredictionEngine(const uniptr<nvinfer1::IBuilder>& builder);
    Dims loadPGMFile(const string& fileName);
    bool validateOutput(int digit);

    template <typename T> uniptr<T> makeunique(T* t)
    {
        if (!t) throw std::runtime_error{"Failed to create TensorRT object"};
        return uniptr<T>{t};
    }

private:
    MNISTParams params;
    nvinfer1::Dims mPredictionInputDims;
    nvinfer1::Dims mPredictionOutputDims;
    uniptr<nvinfer1::ICudaEngine> mPreprocessorEngine{nullptr};
    uniptr<nvinfer1::ICudaEngine> mPredictionEngine{nullptr};
    uniptr<nvinfer1::IExecutionContext> mPreprocessorContext{nullptr};
    uniptr<nvinfer1::IExecutionContext> mPredictionContext{nullptr};
    samplesCommon::ManagedBuffer mInput{};
    samplesCommon::DeviceBuffer mPredictionInput{};
    samplesCommon::ManagedBuffer mOutput{};
};


#endif //TUTO_MNIST_H
