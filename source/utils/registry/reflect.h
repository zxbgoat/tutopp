//
// Created by 张晓彬 on 2021/10/18.
//

#ifndef REFLECT_H
#define REFLECT_H

#include "comm.h"

class Reflector;


// 支持反射的基类
class Base
{
public:
    Base() { cout << "Base()" << endl; }

    virtual ~Base() { cout << "~Base()" << endl; }

    virtual void print() { cout << "Base print()" << endl; }
};


// 工厂类
// - 提供注册接口，注册子类的 Reflector 对象
// - 提供返回创建对象接口，根据子类类名返回实例
class Factory
{
private:
    Factory() { cout << "Factory()" << endl; }

public:
    ~Factory() { cout << "~Factory()" << endl; }

public:
    static Factory& getInstance();
    // 将子类的 Reflector 指针注册到 map 中
    void Register(Reflector* reflector);
    // 根据类名返回实例
    Base* getObject(string className);

private:
    map<string, Reflector*> objectMap;
};


typedef Base* (*ObjectConstructor)();


// 实现反射的类
// - 构造时将自身的(实际上就是子类的) className 和 Reflector写入 map 中
// - 回调函数实现返回子类的实例
class Reflector
{
public:
    Reflector(string name, ObjectConstructor objc) : m_cname(name), m_objc(objc)
    {
        cout << "Reflector()" << endl;
        Factory::getInstance().Register(this);
    }
    virtual ~Reflector(){ cout << "~Reflector()" << endl; }

    Base* getObjectInstance();

public:
    string m_cname;
    ObjectConstructor m_objc;
};


Factory& Factory::getInstance()
{
    static Factory factory;
    return factory;
}

void Factory::Register(Reflector* reflector)
{
    if (reflector)
        objectMap.insert(map<string, Reflector*>::value_type(reflector->m_cname, reflector));
}


Base* Factory::getObject(string className)
{
    map<string, Reflector*>::const_iterator iter = objectMap.find(className);
    if (iter != objectMap.end())
        return iter->second->getObjectInstance();
}


Base* Reflector::getObjectInstance()
{
    return m_objc();
}


#define DELCLARE_CLASS(className)      \
    public:                            \
        static Base* CreateObject()    \
        {                              \
            return new className;      \
        }                              \
    protected:                         \
        static Reflector m_reflector;  \

#define REGISTER_CLASS(className)      \
    Reflector className::m_reflector(#className, className::CreateObject);


#endif
