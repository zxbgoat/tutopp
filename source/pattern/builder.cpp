//
// Created by 张晓彬 on 2021/12/21.
//

#include "comm.h"


class Product1
{
public:
    vector<string> _parts;

    void list_parts() const
    {
        cout << "Product Parts: ";
        for (size_t i=0; i < _parts.size(); ++i)
        {
            if (_parts[i] == _parts.back())
                cout << _parts[i];
            else
                cout << _parts[i] << ", ";
        }
        cout << "\n\n";
    }
};


class Builder
{
public:
    virtual ~Builder() {}
    virtual void ProductPartA() const = 0;
};