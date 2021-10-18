//
// Created by 张晓彬 on 2021/10/11.
//

#include <iostream>
#include <queue>
#include <string>

using std::priority_queue;


struct node
{
    int id;
    int num;
    int next;

    bool operator<(const node &b) const
    {
        if (this->next == b.next)
            return this->id > b.id;
        return this->next > b.next;
    }
};


int main()
{
    string t;
    priority_queue<node> que;
    while (cin >> t)
    {
        if (t == "#") break;
        int id, num;
        cin >> id >> num;
        que.push((node){id, num, num});
    }
    int n;
    cin >> n;
    for (int i = 0; i < n; ++i)
    {
        node tmp = que.top();
        que.pop();
        cout << tmp.id << endl;
        tmp.next += tmp.num;
        que.push(tmp);
    }
    return 0;
}