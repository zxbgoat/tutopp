//
// Created by 张晓彬 on 2021/10/18.
//

#include "comm.h"


class Base
{
public:
    virtual void print() { cout << "Base" << endl; }
};


class Derive1 : public Base
{
public:
    virtual void print() { cout << "Derive1" << endl; }
};


class Derive2 : public Base
{
public:
    virtual void print() { cout << "Derive2" << endl; }
};


class Factory
{
public:
    static Base* build(const string& className)
    {
        if("Derive1" == className) return new Derive1;
        if("Derive2" == className) return new Derive2;
        return 0;
    }
};


int main()
{
    Base* p1 = Factory::build("Derive1");
    p1->print();
    Base* p2 = Factory::build("Derive2");
    p2->print();
    return 0;
}
