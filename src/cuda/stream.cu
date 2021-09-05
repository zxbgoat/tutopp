//
// Created by 张晓彬 on 2021/9/5.
//

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


void create(int size)
{
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i)
        cudaStreamCreate(&stream[i]);
    float* hostptr;
    cudaMallocHost(&hostptr, 2*size);
}