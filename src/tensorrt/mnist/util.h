//
// Created by 张晓彬 on 2021/6/8.
//

#ifndef TUTO_UTIL_H
#define TUTO_UTIL_H

#include "comm.h"
#include "mnist.h"


string joinpath(initializer_list<string> paths)
{
    string result;
    for (auto iter = paths.begin(); iter != paths.end(); ++iter)
    {
        string path = *iter;
        if (iter != paths.end()-1 && path.back() != '/')
            path += "/";
        result += path;
    }
    return result;
}


ostream& operator<<(ostream& out, MNISTParams& params)
{
    out << "===================[Parameters]====================" << endl;
    out << "            application | " << params.applic         << endl;
    out << "            input width | " << params.inw            << endl;
    out << "           input height | " << params.inh            << endl;
    out << "            output size | " << params.outsize        << endl;
    out << "               use int8 | " << params.int8           << endl;
    out << "               use fp16 | " << params.fp16           << endl;
    out << "            weight file | " << params.weightfile     << endl;
    out << "            means proto | " << params.meanfile       << endl;
    out << "             batch size | " << params.batchsize      << endl;
    out << "               dla core | " << params.dlacore        << endl;
    out << "             input name | " << params.inname         << endl;
    out << "            output name | " << params.outname        << endl;
    out << "         data directory | " << params.datadir        << endl;
    out << "===================[Parameters]====================" << endl;
    return out;
}


#endif //TUTO_UTIL_H
