//
// Created by tesla on 2021/6/6.
//

#ifndef TUTO_MNIST_H
#define TUTO_MNIST_H

#include <map>
#include <string>
#include <memory>

#include <buffers.h>
#include <common.h>
#include <argsParser.h>
#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <logger.h>

using std::map;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::ifstream;
using samplesCommon::BufferManager;
using samplesCommon::InferDeleter;

template<typename T>
using uniptr = unique_ptr<T, InferDeleter>;
using wtsmap = map<string, nvinfer1::Weights>;


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


#endif //TUTO_MNIST_H
