//
// Created by tesla on 2021/6/6.
//

#ifndef TUTO_MNIST_H
#define TUTO_MNIST_H

#include "comm.h"

struct MNISTParams
{
    string applic;
    int inh, inw;
    int outsize;
    bool fp16, int8;
    string weightfile;
    string protocfile;
    string meanfile;
    string inname;
    string outname;
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
    shared_ptr<ICudaEngine> engine;

};


class PluginMNIST
{
public:
    PluginMNIST(const MNISTParams& params);
    bool build();
    bool infer();
    bool teardown();

protected:
    void constructNetwork(uniptr<IBuilder>& builder, uniptr<ICaffeParser>& parser, uniptr<INetworkDefinition>& network);
    bool preprocess(const BufferManager& buffers, const string& inname, int digit) const;
    bool postprocess(const BufferManager& buffers, const string& outname, int digit) const;

private:
    shared_ptr<ICudaEngine> engine{nullptr};
    MNISTParams params;
    uniptr<IBinaryProtoBlob> meanblob;
    nvinfer1::Dims indims;
};


class DynamicReshape
{
public:
    explicit DynamicReshape(MNISTParams  params);
    bool build();
    bool prepare();
    bool infer();

private:
    bool buildproc(const uniptr<IBuilder>& builder);
    bool buildpred(const uniptr<IBuilder>& builder);
    Dims preprocess(const string& filepath);
    bool postpreocess(int digit);

private:
    MNISTParams params;
    nvinfer1::Dims predindims;
    nvinfer1::Dims predoutdims;
    uniptr<ICudaEngine> procengine{nullptr};
    uniptr<ICudaEngine> predengine{nullptr};
    uniptr<IExecutionContext> proccontext{nullptr};
    uniptr<IExecutionContext> predcontext{nullptr};
    samplesCommon::ManagedBuffer input{};
    samplesCommon::DeviceBuffer predinput{};
    samplesCommon::ManagedBuffer output{};
};


#endif //TUTO_MNIST_H
