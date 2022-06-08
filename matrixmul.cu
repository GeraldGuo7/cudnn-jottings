//system includes
#include <assert.h>
#include <stdio.h>

//CUDA runtime
#include <cuda_runtime.h>

//Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

//Matrix multiplication on the device: C=A*B, wA
//is A's width and wB is B's width
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B, 
                                                        int wA, int wB)
{
  //Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  //Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  //Index of the first sub-matrix of A processed by the block.
  int aBegin = wA*BLOCK_SIZE*by;
  
  //Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA -1;
  
  //step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;
  
  //Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE*bx;
  
  //step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE*wB;
  
  //Csub is used to store the element of the block sub-matrix that 
  //is computed by the thread
  float Csub = 0;
  
  //Loop over all sub-matrices of A and B required to compute the block
  for(int a=aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep)
  {
    //Declaration of the shared memory array as used to store the submatrix
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    //load the matrices from device memory to shared memory.
    As[ty][tx] = A[a+wA*ty+tx];
    Bs[ty][tx] = B[b+wB*ty+tx];
    
    //Synchronize to make sure the matrices are loaded.
    __syncthreads();
    
    //Multiply the two matrices, each thread computes one element of the block
#pragma unroll
    for(int k=0; k<BLOCK_SIZE; ++k)
    {
      Csub += As[ty][k]*Bs[k][tx];
    }
    __syncthreads();
  }
  int c = wB*BLOCK_SIZE*by + BLOCK_SIZE*bx;
  C[c+wB*ty+tx] = Csub;
}

void ConstantInit(float *data, int size, float val)
{
  for(int i=0; i<size; ++i)
  {
    data[i] = val;
  }
}

int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA,
                   const dim3 &dimsB)
{
  //Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float)*size_A;
  
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  
  unsigned int size_B = dimsB.x *dimsB.y;
  unsigned int mem_size_B = sizeof(float)*size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  cudaStream_t stream;
  
  //Initialize host memory
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);
  
  //Allocate device memory
  float *d_A, *d_B, *d_C;
  
  //Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x *dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
  
  if(h_C == NULL)
  {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }
  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
  //copy host memory to device
  checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  
  //setup execution parameters
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x/threads.x, dimsA.y/threads.y);
  
  //Create and start timer
  printf("COmputing result using CUDA Kernel...\n");
  
  //Performs warmup operation using matrixMul CUDA kernel
  if(block_size == 16)
  {
    MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  } else {
    MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }
  
  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  //Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));
  
  //Execute the kernel
  int nIter = 300;
  for(int j=0; j<nIter; j++)
  {
    if(block_size == 16)
    {
      MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    } else {
      MatrixMulCUDA<32><<<grid, thread, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
  }
  
  //Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));
  
}




