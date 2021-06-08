//
// Created by 张晓彬 on 2021/6/8.
//

#ifndef TUTO_UTIL_H
#define TUTO_UTIL_H

#include "comm.h"


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


#endif //TUTO_UTIL_H
