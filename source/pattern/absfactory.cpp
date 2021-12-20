//
// Created by 张晓彬 on 2021/11/17.
//

#include "comm.h"


class AbstractProductA
{
public:
    virtual ~AbstractProductA() {}
    virtual string UsefulFunctionA() const = 0;
};


class ConcreteProductA1: AbstractProductA
{
public:
    string UsefulFunctionA() const override
    {
        return "The result of the product A1";
    }
};


class ConcreteProductA2: AbstractProductA
{
public:
    string UsefulFunctionA() const override
    {
        return "The result of the product A2";
    }
};


class AbstractProductB
{
public:
    virtual ~AbstractProductB() {}
    virtual string UsefulFunctionB() const = 0;
};
