//
// Created by 张晓彬 on 2021/10/12.
//

#include "comm.h"


class Solution
{
    int singleNumber(vector<int>& nums)
    {
        unordered_set<int> s;
        for (int i = 0; i < nums.size(); ++i)
        {
            if (s.count(nums[i]) == 1)
                s.erase(nums[i]);
            else
                s.insert(nums[i]);
        }
        return *s.begin();
    }
};


class Solution2
{
    /*
     *  0 ^ x ==> x
     *  x ^ x ==> 0
     */
    int singleNumber(vector<int>& nums)
    {
        int ans = 0;
        for (int i = 0; i < nums.size(); ++i)
            ans ^= nums[i];
        return ans;
    }
};
