//
// Created by tesla on 2021/6/6.
//

#include "util.h"
#include "mnist.h"


MNISTParams initparams(const cmdparser& parser)
{
    MNISTParams params;
    params.applic = parser.get<string>("app");
    params.datadir = parser.get<string>("data");
    params.weightfile = parser.get<string>("wts");
    params.protocfile = parser.get<string>("proto");
    params.meanfile = parser.get<string>("mean");
    params.inname = "Input3";
    params.outname = "Plus214_Output_0";
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
    parser.add<string>("app", 'a', "app name", false, "dynashape");
    parser.add<string>("data", 'd', "data dir", false, "data/mnist");
    parser.add<string>("wts", 't', "weights file", false, "mnist.onnx");
    parser.add<string>("proto", 'p', "prototxt file", false, "");
    parser.add<string>("mean", 'm', "mean file", false, "");
    parser.add<int>("out", 'o', "output size", false, 0);
    parser.add<int>("inw", 'w', "input width", false, 28);
    parser.add<int>("inh", 'h', "input height", false, 28);
    parser.add<int>("batch", 'b', "batch size", false, 1);
    parser.add<int>("dla", 'l', "dla core", false, 0);
    parser.add<bool>("fp16", 'f', "run in fp16", false, true);
    parser.add<bool>("int8", 'i', "run in int8", false, true);
    parser.parse(argc, argv);
    MNISTParams params = initparams(parser);
    cout << params << endl;
    auto test = Logger::defineTest("dynareshape", argc, argv);
    Logger::reportTestStart(test);
    DynamicReshape dynareshape(params);
    if (!dynareshape.build())   return gLogger.reportFail(test);
    if (!dynareshape.prepare()) return gLogger.reportFail(test);
    if (!dynareshape.infer())   return gLogger.reportFail(test);
    return gLogger.reportPass(test);
}