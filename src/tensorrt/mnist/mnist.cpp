//
// Created by tesla on 2021/6/6.
//

#include "mnist.h"


MNISTParams initparams(const cmdparser& parser)
{
    MNISTParams params;
    params.datadir = parser.get<string>("data");
    params.weightfile = parser.get<string>("wts");
    params.meansproto = parser.get<string>("proto");
    params.innames.emplace_back("Input3");
    params.outnames.emplace_back("Plus214_Output_0");
    params.batchsize = parser.get<int>("batch");
    params.dlacore = parser.get<int>("dla");
    params.outsize = parser.get<int>("out");
    params.inh = parser.get<int>("inh");
    params.inw = parser.get<int>("inw");
    params.fp16 = parser.get<bool>("fp16");
    params.int8 = parser.get<bool>("int8");
    return params;
}


int main(int argc, char** argv)
{
    cmdparser parser;
    parser.add<string>("app", 'a', "app name", true);
    parser.add<string>("data", 'd', "data dir", false, "data/mnist");
    parser.add<string>("wts", 't', "weights file", false, "mnist.onnx");
    parser.add<string>("proto", 'p', "means proto", false, "");
    parser.add<int>("out", 'o', "output size", false, 0);
    parser.add<int>("inw", 'w', "input width", false, 28);
    parser.add<int>("inh", 'h', "input height", false, 28);
    parser.add<int>("batch", 'b', "batch size", false, 1);
    parser.add<int>("dla", 'l', "dla core", false, 0);
    parser.add<bool>("fp16", 'f', "run in fp16", false, true);
    parser.add<bool>("int8", 'i', "run in int8", false, true);
    parser.parse(argc, argv);
    MNISTParams params = initparams(parser);
    auto test = Logger::defineTest("dynareshape", argc, argv);
    Logger::reportTestStart(test);
    DynamicReshape dynareshape(params);
    if (!dynareshape.build())   return gLogger.reportFail(test);
    if (!dynareshape.prepare()) return gLogger.reportFail(test);
    if (!dynareshape.infer())   return gLogger.reportFail(test);
    return gLogger.reportPass(test);
}