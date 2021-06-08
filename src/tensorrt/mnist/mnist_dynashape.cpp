//
// Created by 张晓彬 on 2021/6/8.
//


#include <utility>

#include "util.h"
#include "mnist.h"


DynamicReshape::DynamicReshape(MNISTParams params) { this->params = move(params); }


bool DynamicReshape::build()
{
    auto builder = uniptr<IBuilder>(createInferBuilder(gLogger.getTRTLogger()));
    buildpred(builder);
    buildproc(builder);
    return true;
}


bool DynamicReshape::buildproc(const uniptr<nvinfer1::IBuilder>& builder)
{
    const auto explibatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = uniptr<INetworkDefinition>(builder->createNetworkV2(explibatch));
    // Reshape a dynamically shaped input to the size expected by the model, (1, 1, 28, 28).
    Dims4 dims{1, 1, -1, -1};
    auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, dims);
    auto resize = network->addResize(*input);
    resize->setOutputDimensions(predindims);
    network->markOutput(*resize->getOutput(0));
    // Finally, configure and build the preprocessor engine.
    auto config = uniptr<IBuilderConfig>(builder->createBuilderConfig());
    // Create an optimization profile so that we can specify a range of input dimensions.
    auto profile = builder->createOptimizationProfile();
    // This profile will be valid for all images whose size falls in the range of
    // [(1, 1, 1, 1), (1, 1, 56, 56)] but TensorRT will optimize for (1, 1, 28, 28)
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, 1, 1, 1});
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, 1, 28, 28});
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{1, 1, 56, 56});
    config->addOptimizationProfile(profile);
    procengine = uniptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    gLogInfo << "Profile dimensions in preprocessor engine:" << endl;
    gLogInfo << "    Minimum = " << procengine->getProfileDimensions(0, 0, OptProfileSelector::kMIN) << endl;
    gLogInfo << "    Optimum = " << procengine->getProfileDimensions(0, 0, OptProfileSelector::kOPT) << endl;
    gLogInfo << "    Maximum = " << procengine->getProfileDimensions(0, 0, OptProfileSelector::kMAX) << endl;
    return true;
}


bool DynamicReshape::buildpred(const uniptr<IBuilder> &builder)
{
    // Create a network using the parser.
    const auto explibatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = uniptr<INetworkDefinition>(builder->createNetworkV2(explibatch));
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    string wtspath = joinpath({params.datadir, params.weightfile});
    bool status = parser->parseFromFile(wtspath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!status) throw std::runtime_error{"Failed to parse model"};
    // Attach a softmax layer to the end of the network.
    auto softmax = network->addSoftMax(*network->getOutput(0));
    // Set softmax axis to 1 since network output has shape [1, 10] in full dims mode
    softmax->setAxes(1 << 1);
    network->unmarkOutput(*network->getOutput(0));
    network->markOutput(*softmax->getOutput(0));
    // Get information about the inputs/outputs directly from the model.
    predindims = network->getInput(0)->getDimensions();
    predoutdims = network->getOutput(0)->getDimensions();
    // Create a builder config
    auto config = uniptr<IBuilderConfig>(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(16_MiB);
    if (params.fp16) config->setFlag(BuilderFlag::kFP16);
    if (params.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
    predengine = uniptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
}


bool DynamicReshape::prepare()
{
    proccontext = uniptr<IExecutionContext>(procengine->createExecutionContext());
    predcontext = uniptr<IExecutionContext>(predengine->createExecutionContext());
    predinput.resize(predindims);
    output.hostBuffer.resize(predoutdims);
    output.deviceBuffer.resize(predoutdims);
}


bool DynamicReshape::infer()
{
    random_device rd{};
    default_random_engine generator{rd()};
    uniform_int_distribution<int> digitDistribution{0, 9};
    int digit = digitDistribution(generator);
    string impath = joinpath({params.datadir, to_string(digit)+".pgm"});
    Dims indims = preprocess(impath);
    input.deviceBuffer.resize(indims);
    cudaMemcpy(input.deviceBuffer.data(), input.hostBuffer.data(),
               input.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
    // Set the input size for the preprocessor
    proccontext->setBindingDimensions(0, indims);
    // We can only run inference once all dynamic input shapes have been specified.
    if (!proccontext->allInputDimensionsSpecified()) return false;
    // Run the preprocessor to resize the input to the correct shape
    vector<void*> procbinds = {input.deviceBuffer.data(), predinput.data()};
    // For engines using full dims, we can use executeV2,
    // which does not include a separate batch size parameter.
    if (!proccontext->executeV2(procbinds.data())) return false;
    // Next, run the model to generate a prediction.
    vector<void*> predbinds = {predinput.data(), output.deviceBuffer.data()};
    if (!predcontext->executeV2(predbinds.data())) return false;
    // Copy the outputs back to the host and verify the output.
    cudaMemcpy(output.hostBuffer.data(), output.deviceBuffer.data(),
               output.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost);
    return postpreocess(digit);
}

Dims DynamicReshape::preprocess(const string &filepath)
{
    ifstream infile(filepath, ifstream::binary);
    string magic;
    int h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    Dims4 indims{1, 1, h, w};
    size_t vol = samplesCommon::volume(indims);
    vector<uint8_t> fileData(vol);
    infile.read(reinterpret_cast<char*>(fileData.data()), vol);
    gLogInfo << "Input:\n";
    for (size_t i = 0; i < vol; i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % w) ? "" : "\n");
    gLogInfo << std::endl;
    // Normalize and copy to the host buffer.
    input.hostBuffer.resize(indims);
    float* hostDataBuffer = static_cast<float*>(input.hostBuffer.data());
    transform(fileData.begin(), fileData.end(), hostDataBuffer,
              [](uint8_t x) { return 1.0 - static_cast<float>(x / 255.0); });
    return indims;
}


bool DynamicReshape::postpreocess(int digit)
{
    const auto* rawbuf = static_cast<const float*>(output.hostBuffer.data());
    vector<float> prob(rawbuf, rawbuf+output.hostBuffer.size());
    int idx{0};
    for (const auto& elem : prob)
    {
        gLogInfo << " Prob " << idx << "  " << fixed << setw(5) << setprecision(4) << elem << " "
                 << "Class " << idx << ": " << string(int(floor(elem*10+0.5f)), '*') << endl;
        ++idx;
    }
    int predict = max_element(prob.begin(), prob.end())-prob.begin();
    return digit == predict;
}
