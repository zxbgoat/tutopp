//
// Created by tesla on 2021/6/6.
//

#include "util.h"
#include "mnist.h"
#include "factory.h"


PluginMNIST::PluginMNIST(const MNISTParams &params) { this->params = params; }


bool PluginMNIST::build()
{
    auto builder = uniptr<IBuilder>(createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) return false;
    auto network = uniptr<INetworkDefinition>(builder->createNetwork());
    if (!network) return false;
    auto config = uniptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;
    auto parser = uniptr<ICaffeParser>(createCaffeParser());
    if (!parser) return false;
    // The PluginFactory object contains the methods to construct the FC plugin layer
    // that are needed to create the engine
    PluginFactory parserPluginFactory;
    parser->setPluginFactoryExt(&parserPluginFactory);
    constructNetwork(builder, parser, network);
    builder->setMaxBatchSize(params.batchsize);
    config->setMaxWorkspaceSize(1_MiB);
    if (params.fp16) config->setFlag(BuilderFlag::kFP16);
    if (params.int8) config->setFlag(BuilderFlag::kINT8);
    samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    samplesCommon::enableDLA(builder.get(), config.get(), params.dlacore);
    // For illustrative purposes, we will use the builder to create a CUDA engine,
    // serialize it to mModelStream object (which can be written to a file), then
    // deserialize mModelStream with a IRuntime object to recreate the original engine.
    // Note for this sample we could have simply used the original engine produced by builder->buildEngineWithConfig()
    auto modelStream = uniptr<IHostMemory>(builder->buildEngineWithConfig(*network, *config)->serialize());
    assert(modelStream != nullptr);
    auto runtime = uniptr<IRuntime>(createInferRuntime(gLogger.getTRTLogger()));
    if (params.dlacore >= 0) runtime->setDLACore(params.dlacore);

    // The PluginFactory object also contains the methods needed to deserialize
    // our engine that was built with the FC plugin layer
    PluginFactory pluginFactory;
    engine = shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), &pluginFactory), InferDeleter());
    gLogInfo << "Done preparing engine..." << std::endl;
    assert(network->getNbInputs() == 1);
    indims = network->getInput(0)->getDimensions();
    assert(indims.nbDims == 3);
    return true;
}


void PluginMNIST::constructNetwork(uniptr<IBuilder>& builder, uniptr<ICaffeParser>& parser, uniptr<INetworkDefinition>& network)
{
    auto type = builder->platformHasFastFp16() ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    const IBlobNameToTensor* blob2tensor = parser->parse(params.protocfile.c_str(), params.weightfile.c_str(), *network, type);
    network->markOutput(*blob2tensor->find(params.outname.c_str()));
    // Parse mean blob for preprocessing input later
    meanblob = uniptr<IBinaryProtoBlob>(parser->parseBinaryProto(params.meanfile.c_str()));
    gLogInfo << "Done constructing network..." << endl;
}


bool PluginMNIST::infer()
{
    // Create RAII buffer manager object
    BufferManager buffers(engine, params.batchsize);
    auto context = uniptr<IExecutionContext>(engine->createExecutionContext());
    if (!context) return false;
    // Pick a random digit to try to infer
    srand(time(NULL));
    const int digit = rand() % 10;
    // Read the input data into the managed buffers
    // There should be just 1 input tensor
    if (!preprocess(buffers, params.inname, digit)) return false;
    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);
    // Asynchronously enqueue the inference work
    if (!context->enqueue(params.batchsize, buffers.getDeviceBindings().data(), stream, nullptr))
        return false;
    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);
    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);
    // Release stream
    cudaStreamDestroy(stream);
    // Check and print the output of the inference. There should be just one output tensor
    bool correct = postprocess(buffers, params.outname, digit);
    // The output correctness is not used to determine the test result.
    if (!correct && params.dlacore != -1)
        gLogInfo << "Warning: infer result is not correct. It maybe caused by dummy scales in INT8 mode." << endl;
    return true;
}


bool PluginMNIST::preprocess(const BufferManager& buffers, const string& inname, int digit) const
{
    const int inputH = indims.d[1];
    const int inputW = indims.d[2];
    // Read a random digit file
    srand(unsigned(time(nullptr)));
    vector<uint8_t> fileData(inputH * inputW);
    string impath = joinpath({params.datadir, to_string(digit)+".pgm"})
    readPGMFile(impath, fileData.data(), inputH, inputW);
    // Print ASCII representation of digit
    gLogInfo << "Input:\n";
    for (int i = 0; i < inputH * inputW; i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    gLogInfo << endl;
    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inname));
    const float* meandata = reinterpret_cast<const float*>(meanblob->getData());
    for (int i = 0; i < inputH * inputW; i++)
        hostInputBuffer[i] = static_cast<float>(fileData[i]) - meandata[i];
    return true;
}


bool PluginMNIST::postprocess(const BufferManager &buffers, const string &outname, int digit) const
{
    const auto* prob = static_cast<const float*>(buffers.getHostBuffer(outname));
    // Print histogram of the output distribution
    gLogInfo << "Output:\n";
    float val{0.0f};
    int idx{0};
    const int kDIGITS = 10;
    for (int i = 0; i < kDIGITS; i++)
    {
        if (val < prob[i]) { val = prob[i]; idx = i; }
        gLogInfo << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    gLogInfo << std::endl;
    return (idx == digit && val > 0.9f);
}


bool PluginMNIST::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}
