//
// Created by 张晓彬 on 2021/11/17.
//

#include <string>
#include <iostream>

using std::cout;
using std::string;


class AbstractProductA
{
public:
    virtual ~AbstractProductA() {};
    virtual string UsefulFunctionA() const = 0;
};


class ConcreteProductA1: AbstractProductA
{
public:
    string UsefulFunctionA() const override
    {
        return "The result of product A1";
    }
};


class ConcreteProductA2: public AbstractProductA
{
    string UsefulFunctionA() const override
    {
        return "The result of the product A2";
    }
};
