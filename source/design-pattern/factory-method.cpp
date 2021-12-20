//
// Created by 张晓彬 on 2021/11/17.
//

#include <string>
#include <iostream>

using std::cout;
using std::string;


class Product
{
public:
    virtual ~Product();
    virtual string Operation() const = 0;
};


class ConcreteProduct1: public Product
{
public:
    string Operation() const override
    {
        return "{Result of the ConcreteProduct1}";
    }
};


class ConcreteProduct2: public Product
{
public:
    string Operation() const override
    {
        return "{Result of the ConcreteProduct2}";
    }
};


class Creator
{
public:
    virtual ~Creator() {}
    virtual Product* FactoryMethod() const = 0;

    string SomeOperation() const
    {
        Product* product = this->FactoryMethod();
        string result = "Creator: The same creator's code has just worked with " + product->Operation();
        delete product;
        return result;
    }
};


class ConcreteCreator1: public Creator
{
public:
    Product* FactoryMethod() const override
    {
        return new ConcreteProduct1();
    }
};
