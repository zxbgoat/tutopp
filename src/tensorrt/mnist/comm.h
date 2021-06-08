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
using std::unique_ptr;
using std::shared_ptr;
using std::ifstream;
using std::initializer_list;

using samplesCommon::BufferManager;
using samplesCommon::InferDeleter;
using samplesCommon::OnnxSampleParams;
using cmdparser = cmdline::parser;

template<typename T>
using uniptr = unique_ptr<T, InferDeleter>;
using wtsmap = map<string, nvinfer1::Weights>;

#endif //TUTO_COMM_H
