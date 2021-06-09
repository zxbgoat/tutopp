//
// Created by tesla on 2021/6/9.
//

#include <stdio.h>

#ifdef OPEN_LOG

#define LOG(frm, args...)                    \
{                                            \
    printf("[%s : %d]", __FILE__, __LINE__); \
    printf(frm, ##args);                       \
    printf("\n");                            \
}

#else

#define LOG(frm, args...)

#endif

int main()
{
    int a = 1, b = 2;
    LOG("a = %d, b = %d", a, b);
    return 0;
}