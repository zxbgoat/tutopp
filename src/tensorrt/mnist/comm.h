//
// Created by 张晓彬 on 2021/6/8.
//

#ifndef TUTO_COMM_H
#define TUTO_COMM_H

#include <map>
#include <string>
#include <memory>
#include <utility>
#include <random>
#include <initializer_list>

#include <common.h>
#include <logger.h>
#include <buffers.h>
#include <NvInfer.h>
#include <argsParser.h>
#include <NvCaffeParser.h>
#include <parserOnnxConfig.h>
#include <cuda_runtime_api.h>
#include <cmd/cmd.h>

using std::map;
using std::endl;
using std::cout;
using std::string;
using std::ifstream;
using std::unique_ptr;
using std::shared_ptr;
using std::initializer_list;
using std::random_device;
using std::default_random_engine;
using std::uniform_int_distribution;
using std::transform;
using std::fixed;
using std::setw;
using std::setprecision;
using std::floor;
using std::max_element;
using std::move;

using samplesCommon::BufferManager;
using samplesCommon::InferDeleter;
using samplesCommon::OnnxSampleParams;

using nvinfer1::IBuilder;
using nvinfer1::createInferBuilder;
using nvinfer1::INetworkDefinition;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IHostMemory;
using nvinfer1::IRuntime;
using nvinfer1::createInferRuntime;

using nvcaffeparser1::ICaffeParser;
using nvcaffeparser1::IBinaryProtoBlob;
using nvcaffeparser1::createCaffeParser;
using nvcaffeparser1::IBlobNameToTensor;

using cmdparser = cmdline::parser;

template<typename T>
using uniptr = unique_ptr<T, InferDeleter>;
using wtsmap = map<string, nvinfer1::Weights>;

#endif //TUTO_COMM_H
