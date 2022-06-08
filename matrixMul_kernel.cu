#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#define AS(i,j) As[i][j]
#define BS(i,j) Bs[i][j]

template <int block_size, typename size_type> __device__ void matrixMul(float *C, float *A,
                                                                        float *B, size_type wA,
                                                                        size_type wB)
{
  //Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;
  
  //Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;
  
  //Index of the first submatrix of A processed by the block
  size_type aBegin = wA*block_size *by;
  
  //Index of the last submatrix of A processed by the block
  size_type aEnd = aBegin +wA - 1;
  
  //step size used to iterate through the submatrix of A
  size_type aStep = block_size;
  
  //index of the first submatrix of B processed by the block
  size_type bBegin = block_size *bx;
  
  //step size used to iterate through the submatrices of B
  size_type bStep = block_size*wB;
  
  //Csub is used to store the element of the block submatrix that
  //is computed by the thread
  float Csub = 0;
  
  //loop over all the submatrices of A and B required to compute the block submatrix
  for(size_type a = aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep)
  {
    //Declaration of the shared memory array As used to store the submatrix of A.
    __shared__ float As[block_size][block_size];
    //Declaration of the shared memory array Bs used to store the submatrix of B.
    __shared__ float Bs[block_size][block_size];
    
    //load the matrices from device memory to shared memory 
    AS(ty, tx) = A[a+wA*ty+tx];
    BS(ty, tx) = B[b+wB*ty+tx];
    
    //synchronize to make sure the matrices are loaded.
    __syncthreads();
#pragma unroll
    for(size_type k=0; k<block_size; ++k)
    {
      Csub += AS(ty,k)*BS(k, tx);
    }
    __syncthreads();
  }
  size_type c = wB*block_size*by + block_size*bx;
  C[c+wB*ty+tx] = Csub;
}

//C wrappers around our template kernel
extern "C" __global__ void matrixMul_bs16_64bit(float *C, float *A, float *B, 
                                                size_t wA, size_t wB)
{
  matrixMul<16, size_t>(C, A, B, wA, wB);
}

extern "C" __global__ void matrixMul_bs32_64bit(float *C, float *A, float *B,
                                                size_t wA, size_t wB)
{
  matrixMul<32, size_t>(C, A, B, wA, wB);
}
#endif







