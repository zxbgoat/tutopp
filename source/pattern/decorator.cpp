//
// Created by 张晓彬 on 2021/12/21.
//

#include "comm.h"

class Component
{
public:
    virtual ~Component() {}
    virtual string Operation() const = 0;
};


class ConcreteComponent: public Component
{
public:
    string Operation() const override
    {
        return "ConcreteComponent";
    }
};


class Decorator: public Component
{
public:
    Decorator(Component* component): component(component) {}

    string Operation() const override { return component->Operation(); }

protected:
    Component* component;
};


class ConcreteDecoratorA: public Decorator
{
public:
    ConcreteDecoratorA(Component* component): Decorator(component) {}

    string Operation() const override
    {
        return "ConcreteDecoratorA(" + Decorator::Operation() + ")";
    }
};


class ConcreteDecoratorB: public Decorator
{
public:
    ConcreteDecoratorB(Component* component): Decorator(component) {}

    string Operation() const override
    {
        return "ConcreteDecoratorB(" + Decorator::Operation() + ")";
    }
};


void client_code(Component* component)
{
    cout << "RESULT: " << component->Operation();
}


int main()
{
    Component* simple = new ConcreteComponent;
    cout << "Client: I've got a simple component:\n";
    client_code(simple);
    Component* decorator1 = new ConcreteDecoratorA(simple);
    Component* decorator2 = new ConcreteDecoratorB(decorator1);
    cout << "Client: Now I've got a decorated component:\n";
    client_code(decorator2);
    cout << "\n";
    delete simple;
    delete decorator1;
    delete decorator2;
    return 0;
}
