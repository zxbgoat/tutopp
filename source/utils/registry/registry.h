//
// Created by 张晓彬 on 2021/10/19.
//

#ifndef TUTO_REGISTRY_H
#define TUTO_REGISTRY_H

#include "comm.h"


// 支持反射的基类
class Base
{
public:
    Base() { cout << "Base()" << endl; }

    virtual ~Base() { cout << "~Base()" << endl; }

    virtual void print() { cout << "Base print()" << endl; }
};


typedef Base* (*ObjectConstructor)();


// 工厂类
// - 提供注册接口，注册子类的 Reflector 对象
// - 提供返回创建对象接口，根据子类类名返回实例
class Factory
{
private:
    Factory(){ cout << "Factory()" << endl; }

public:
    ~Factory(){ cout << "~Factory()" << endl; }

public:
    static Factory& getInstance();

    void Register(string className, ObjectConstructor objc);

    Base* getObject(string className);

private:
    map<string, ObjectConstructor> objectMap;
};


// impl
Factory& Factory::getInstance()
{
    static Factory factory;
    return factory;
}

void Factory::Register(string className, ObjectConstructor m_objc)
{
    if (m_objc)
    {
        objectMap.insert(map<string, ObjectConstructor>::value_type(className, m_objc));
    }
}

Base* Factory::getObject(string className)
{
    map<string, ObjectConstructor>::const_iterator iter = objectMap.find(className);
    if (iter != objectMap.end())
    {
        return iter->second();
    }
}


// 实现反射的类
// - 构造时将自身的(实际上就是子类的) className 和 Reflector写入 map 中
// - 回调函数实现返回子类的实例
class Reflector
{
public:
    Reflector(string name, ObjectConstructor objc)
    {
        cout << "Reflector()" << endl;
        Factory::getInstance().Register(name, objc);
    }
    virtual ~Reflector(){ cout << "~Reflector()" << endl; }

};


#define REGISTER(className)                                               \
    Base* CreateObject##className()                                       \
    {                                                                     \
        return new className;                                             \
    }                                                                     \
    Reflector reflector##className(#className, CreateObject##className);  \


#endif  // TUTO_REGISTRY_H