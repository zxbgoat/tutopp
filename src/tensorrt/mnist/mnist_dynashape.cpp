//
// Created by 张晓彬 on 2021/6/8.
//


#include "util.h"
#include "mnist.h"


bool DynamicReshape::build()
{
    auto builder = makeunique(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    buildPreprocessorEngine(builder);
    buildPredictionEngine(builder);
    return true;
}


bool DynamicReshape::buildPreprocessorEngine(const uniptr<nvinfer1::IBuilder>& builder)
{
    auto preprocessorNetwork = makeunique(builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    // Reshape a dynamically shaped input to the size expected by the model, (1, 1, 28, 28).
    auto input = preprocessorNetwork->addInput("input", nvinfer1::DataType::kFLOAT, Dims4{1, 1, -1, -1});
    auto resizeLayer = preprocessorNetwork->addResize(*input);
    resizeLayer->setOutputDimensions(mPredictionInputDims);
    preprocessorNetwork->markOutput(*resizeLayer->getOutput(0));
    // Finally, configure and build the preprocessor engine.
    auto preprocessorConfig = makeunique(builder->createBuilderConfig());
    // Create an optimization profile so that we can specify a range of input dimensions.
    auto profile = builder->createOptimizationProfile();
    // This profile will be valid for all images whose size falls in the range of [(1, 1, 1, 1), (1, 1, 56, 56)]
    // but TensorRT will optimize for (1, 1, 28, 28)
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, 1, 1, 1});
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, 1, 28, 28});
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{1, 1, 56, 56});
    preprocessorConfig->addOptimizationProfile(profile);
    mPreprocessorEngine = makeunique(builder->buildEngineWithConfig(*preprocessorNetwork, *preprocessorConfig));
    gLogInfo << "Profile dimensions in preprocessor engine:" << endl;
    gLogInfo << "    Minimum = " << mPreprocessorEngine->getProfileDimensions(0, 0, OptProfileSelector::kMIN) << endl;
    gLogInfo << "    Optimum = " << mPreprocessorEngine->getProfileDimensions(0, 0, OptProfileSelector::kOPT) << endl;
    gLogInfo << "    Maximum = " << mPreprocessorEngine->getProfileDimensions(0, 0, OptProfileSelector::kMAX) << endl;
    return true;
}


bool DynamicReshape::buildPredictionEngine(const uniptr<nvinfer1::IBuilder>& builder)
{
    // Create a network using the parser.
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = makeunique(builder->createNetworkV2(explicitBatch));
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    string wtspath = joinpath({params.datadir, params.weightfile});
    bool parsingSuccess = parser->parseFromFile(wtspath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsingSuccess) throw std::runtime_error{"Failed to parse model"};
    // Attach a softmax layer to the end of the network.
    auto softmax = network->addSoftMax(*network->getOutput(0));
    // Set softmax axis to 1 since network output has shape [1, 10] in full dims mode
    softmax->setAxes(1 << 1);
    network->unmarkOutput(*network->getOutput(0));
    network->markOutput(*softmax->getOutput(0));
    // Get information about the inputs/outputs directly from the model.
    mPredictionInputDims = network->getInput(0)->getDimensions();
    mPredictionOutputDims = network->getOutput(0)->getDimensions();
    // Create a builder config
    auto config = makeunique(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(16_MiB);
    if (params.fp16) config->setFlag(BuilderFlag::kFP16);
    if (params.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    mPredictionEngine = makeunique(builder->buildEngineWithConfig(*network, *config));
}


bool DynamicReshape::prepare()
{
    mPreprocessorContext = makeunique(mPreprocessorEngine->createExecutionContext());
    mPredictionContext = makeunique(mPredictionEngine->createExecutionContext());
    mPredictionInput.resize(mPredictionInputDims);
    mOutput.hostBuffer.resize(mPredictionOutputDims);
    mOutput.deviceBuffer.resize(mPredictionOutputDims);
}


bool DynamicReshape::infer()
{
    std::random_device rd{};
    std::default_random_engine generator{rd()};
    std::uniform_int_distribution<int> digitDistribution{0, 9};
    int digit = digitDistribution(generator);
    Dims inputDims = loadPGMFile(joinpath({params.datadir, to_string(digit)+".pgm"}));
    mInput.deviceBuffer.resize(inputDims);
    CHECK(cudaMemcpy(mInput.deviceBuffer.data(), mInput.hostBuffer.data(), mInput.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));
    // Set the input size for the preprocessor
    mPreprocessorContext->setBindingDimensions(0, inputDims);
    // We can only run inference once all dynamic input shapes have been specified.
    if (!mPreprocessorContext->allInputDimensionsSpecified()) return false;
    // Run the preprocessor to resize the input to the correct shape
    std::vector<void*> preprocessorBindings = {mInput.deviceBuffer.data(), mPredictionInput.data()};
    // For engines using full dims, we can use executeV2, which does not include a separate batch size parameter.
    bool status = mPreprocessorContext->executeV2(preprocessorBindings.data());
    if (!status) return false;
    // Next, run the model to generate a prediction.
    std::vector<void*> predicitonBindings = {mPredictionInput.data(), mOutput.deviceBuffer.data()};
    status = mPredictionContext->executeV2(predicitonBindings.data());
    if (!status) return false;
    // Copy the outputs back to the host and verify the output.
    CHECK(cudaMemcpy(mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));
    return validateOutput(digit);
}

Dims DynamicReshape::loadPGMFile(const string& fileName)
{
    ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    string magic;
    int h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    Dims4 inputDims{1, 1, h, w};
    size_t vol = samplesCommon::volume(inputDims);
    vector<uint8_t> fileData(vol);
    infile.read(reinterpret_cast<char*>(fileData.data()), vol);
    gLogInfo << "Input:\n";
    for (size_t i = 0; i < vol; i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % w) ? "" : "\n");
    gLogInfo << std::endl;
    // Normalize and copy to the host buffer.
    mInput.hostBuffer.resize(inputDims);
    float* hostDataBuffer = static_cast<float*>(mInput.hostBuffer.data());
    std::transform(fileData.begin(), fileData.end(), hostDataBuffer, [](uint8_t x) { return 1.0 - static_cast<float>(x / 255.0); });
    return inputDims;
}


bool DynamicReshape::validateOutput(int digit)
{
    const float* bufRaw = static_cast<const float*>(mOutput.hostBuffer.data());
    std::vector<float> prob(bufRaw, bufRaw + mOutput.hostBuffer.size());

    int curIndex{0};
    for (const auto& elem : prob)
    {
        gLogInfo << " Prob " << curIndex << "  " << std::fixed << std::setw(5) << std::setprecision(4) << elem << " "
                 << "Class " << curIndex << ": " << std::string(int(std::floor(elem*10+0.5f)), '*') << endl;
        ++curIndex;
    }

    int predictedDigit = std::max_element(prob.begin(), prob.end()) - prob.begin();
    return digit == predictedDigit;
}
