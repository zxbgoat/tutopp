//
// Created by 张晓彬 on 2021/10/18.
//

#include "reflect.h"

// 支持反射的子类
class Derive1 : public Base
{
DELCLARE_CLASS(Derive1);

public:
    Derive1() { cout << "Derive()" << endl; }

    virtual ~Derive1() { cout << "~Derive()" << endl; }

    void print() { cout << "Derive1 print()" << endl; }
};
REGISTER_CLASS(Derive1);


class Derive2 : public Base
{
DELCLARE_CLASS(Derive2);

public:
    void print() { cout << "Derive2 print()" << endl; }
};
REGISTER_CLASS(Derive2);


int main()
{
    Base* p1 = Factory::getInstance().getObject("Derive1");
    if (p1) p1->print();
    delete p1;

    // Base* p2 = Factory::getInstance().getObject("Derive2");
    // if (p2) p2->print();
    // delete p2;

    return 0;
}
