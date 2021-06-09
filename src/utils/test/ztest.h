//
// Created by tesla on 2021/6/9.
//

#ifndef TUTO_ZTEST_H
#define TUTO_ZTEST_H

#define EXPECT(a, cmp, b) \
{                         \
   if (!((a) cmp (b)))    \
       printf("Error\n"); \
}

#define EXPECT_EQ(a, b) EXPECT(a, ==, b)
#define EXPECT_NE(a, b) EXPECT(a, !=, b)
#define EXPECT_LT(a, b) EXPECT(a, <, b)
#define EXPECT_LE(a, b) EXPECT(a, <=, b)
#define EXPECT_GT(a, b) EXPECT(a, >, b)
#define EXPECT_GE(a, b) EXPECT(a, ==, b)

#endif //TUTO_ZTEST_H
