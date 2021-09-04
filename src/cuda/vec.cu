//
// Created by 张晓彬 on 2021/9/4.
//


#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


__global__ void mykernel2d(float* devptr, size_t pitch, int wid, int hei)
{
    for (int r = 0; r < hei; ++r)
    {
        float* row = (float*)((char*)devptr + r * pitch);
        for (int c = 0; c < wid; ++c)
            float elem = row[c];
    }
}


void alloc2d()
{
    int wid = 64, hei = 64;
    float* devptr;
    size_t pitch;
    cudaMallocPitch(&devptr, &pitch, wid*sizeof(float), hei);
    mykernel2d<<<100, 512>>>(devptr, pitch, wid, hei);
}


__global__ void mykernel3d(cudaPitchedPtr devpitptr, int wid, int hei, int dep)
{
    char* devptr = devpitptr.ptr;
    size_t pitch = devpitptr.pitch;
    size_t slicepit = pitch * hei;
    for (int z = 0; z < dep; ++z)
    {
        char* slice = devptr + z * slicepit;
        for (int y = 0; y < hei; ++i)
        {
            float* row = (float*)(slice + y*pitch);
            for (int x = 0; x < wid; ++x)
                float elem = row[x];
        }
    }
}


void alloc3d()
{
    int wid = 64, hei = 64, dep = 64;
    cudaExtent extent = make_cudaExtent(wid*sizeof(float), hei, dep);
    cudaPitchedPtr devpitptr;
    cudaMalloc3D(&devpitptr, extent);
    mykernel3d<<<100, 512>>>(devpitptr, wid, hei, dep);
}


void gloval()
{
    __constant__ float constdata[256];
    float data[256];
    cudaMemcpyToSymbol(constdata, data, sizeof(data));
    cudaMemcpyFromSymbol(data, constdata, sizeof(data));

    __device__ float devdata;
    float value = 3.14f;
    cudaMemcpyToSymbol(devdata, &val, sizeof(float));
}