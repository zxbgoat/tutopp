//
// Created by 张晓彬 on 2021/10/18.
//

/*
 * 通过类的名称字符串来生成类的对象，即 ClassXX* object=new "ClassXX"；
 * 可以通过反射来解决这个问题，所谓反射，是程序可以访问、检测和修改它本身状态或行为的一种能力。
 * 设计思路为：
 *   1 为需要反射的类中定义一个创建该类对象的一个回调函数；
 *   2 设计一个工厂类，包含一个map，保存类名和创建实例的回调函数，通过类工厂来动态创建类对象；
 *   3 程序开始运行时，将回调函数存入map（哈希表）里面，类名字做为map的key值。
 */


// 1 定义一个函数指针类型，用于指向创建类实例的回调函数
typedef void* (*PTRCreateObject)();
typedef clsmap map<string, PTRCreateObject>;


// 2 定义和实现一个工厂类，用于保存保存类名和创建类实例的回调函数。
//   工厂类的作用仅仅是用来保存类名与创建类实例的回调函数，所以程序的整个生命周期内，
//   无需多个工厂类的实例，所以这里采用单例模式来涉及工厂类。
class ClassFactory
{
public:
    void* getClassByName(string classname)
    {
        clsmap::const_iterator iter;
        iter = classMap.find(classname);
        if (iter == classMap.end())
            return nullptr;
        else
            return iter->second;
    }

    void reigsterClass(string classname, PTRCreateObject method)
    {
        classMap.insert(pair<string, PTRCreateObject>(name, method));
    }

    static ClassFactory& getInstance()
    {
        static ClassFactory lofactory;
        return lofactory;
    }

private:
    ClassFactory() {};

    map<string, PTRCreateObject> classMap;
};


// 3 将定义的类注册到工厂类中，即将类名称字符串和创建类实例的回调函数保存到工厂类的map中。
//   这里又需要完成两个工作：
//     (1) 定义一个创建类实例的回调函数；
//     (2) 将类名称字符串和我们定义的回调函数保存到工厂类的map中。
//   假设定义了一个TestClassA。
class TestClassA
{
public:
    void print()
    {
        cout << "Hello TestClassA" << endl;
    }
};


TestClassA* createObjectTestClassA
{
    return new TestClassA;
};