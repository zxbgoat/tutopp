//
// Created by 张晓彬 on 2021/10/18.
//

#include "registry.h"


// 支持反射的子类
class DeriveA : public Base
{
public:
    DeriveA(){ cout << "Derive()" << endl; }
    virtual ~DeriveA(){ cout << "~Derive()" << endl; }

    void print()
    {
        cout << "DeriveA print()" << endl;
    }
};
REGISTER(DeriveA);


class DeriveB : public Base
{
public:
    void print()
    {
        cout << "DeriveB print()" << endl;
    }
};
REGISTER(DeriveB);

int main()
{
    Base* p1 = Factory::getInstance().getObject("DeriveA");
    if (p1) p1->print();
    delete p1;

    Base* p2 = Factory::getInstance().getObject("DeriveB");
    if (p2) p2->print();
    delete p2;

    return 0;
}
