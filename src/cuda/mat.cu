//
// Created by 张晓彬 on 2021/9/4.
//

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 16


typedef struct
{
    int width;
    int height;
    float* elements;
} Matrix;


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    float cval = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx,x;
    for (int e = 0; e < A.width; ++e)
        cval += A.elements[row+A.width+e] * B.elements[e*B.width+col];
    C.elements[row*C.width+col] = cval;
}


void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}