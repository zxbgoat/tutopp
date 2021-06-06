//
// Created by tesla on 2021/6/6.
//

#include "mnist.h"


ScratchMNIST::ScratchMNIST(const MNISTParams &params)
{
    this->params = params;
    this->number = 0;
    this->engine = nullptr;
}


bool ScratchMNIST::loadWeights(const string& filepath)
{
    gLogInfo << "Loading weights from " << filepath << endl;
    ifstream in(filepath, std::ios::binary);
}