//matrix multiplication: C=A*B.
//System includes
#include <stdio.h>
#include <assert.h>

//CUDA runtime
#include <cuda_runtime.h>
#include <cuda/pipeline>

#if __CUDA_ARCH__ >= 700
#include <cuda/barrier>
#endif

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <helper_functions.h>
#include <helper_cuda.h>

enum kernels
{
  AsyncCopyMultiStageLargeChunk = 0,
  AsyncCopyLargeChunk = 1,
  AsyncCopyLargeChunkAWBarrier = 2,
  AsyncCopyMultiStageSharedState = 3,
  AsyncCopyMultiStage = 4,
  AsyncCopySingleStage = 5,
  Naive = 6,
  NaiveLargeChunk = 7
};

const char* kernelNames[] = {"AsyncCopyMultiStageLargeChunk", "AsyncCopyLargeChunk", "AsyncCopyMultiStageSharedState"};

constexpr int blockSize = 16;

//multi stage memcpy_async pipeline with large chunk copy
template <int BLOCK_SIZE> __global__ void MatrixMulAsyncCopyMultiStageLargeChunk(float* __restrict__ C,
                                                                                 const float* __restrict__ A,
                                                                                 const float* __restrict__ B,
                                                                                 int wA, int wB)
{
    //Requires BLOCK_SIZE % 4 == 0
    //multi-stage pipeline version
  
    constexpr size_t maxPipelineStages = 4;
    
    //Declaration of the shared memory array As used to store the sub-matrix of A for each stage.
    __shared__ alignas(alignof(float4)) float As[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];
    //Declaration of the shared memory array Bs used to store the sub-matrix of B for each stage.
    __shared__ alignas(alignof(float4)) float Bs[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];
  
    float Csub = 0.0;
    //Index of the first sub-matrix of A processed by the block
    const int aBegin = wA*(BLOCK_SIZE) * blockIdx.y;
  
    //Index of the last sub-matrix of A processed by the block
    const int aEnd = aBegin + wA - 1;
  
    //Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
  
    //Index of the first sub-matrix of B processed by the block
    const in bBegin = BLOCK_SIZE * blockIdx.x;
  
    //Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE*wB;
  
    const int t4x = threadIdx.x * 4;
    const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline():
  
    //loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for(int a = aBegin, b = bBegin, i=0, aStage = aBegin, bStage = bBegin, iStage = 0; a<=aEnd; a+= aStep, b+=bStep, ++i)
    {
        //load the matrices from device memory to shared memory; each thread loads one element of each matrix
        for(; aStage<=a+aStep*maxPipelineStage; aStage+=aStep, bStage += bStep, ++iStage)
        {
            pipe.producer_acquire();
            if( aStage <= aEnd&& t4x < BLOCK_SIZE)
            {
                //Rotating buffer
                const int j= iStage % maxPipelineStages;
                cuda::memcpy_async(&As[j][threadIdx.y][t4x], &A[aStage + wA*threadIdx.y + t4x], shape4, pipe);
                cuda::memcpy_async(&Bs[j][threadIdx.y][t4x], &B[aStage + wA*threadIdx.y + t4x], shape4, pipe);
            }
            pipe.producer_commit();
        }
      
        pipe.consumer_wait();
        //Synchronize to make sure the matrices are loaded
        __syncthreads();
      
        //Rotating buffer
        const int j = i%maxPipelineStages;
      
        //Multiply the two matrices; each thread computes one element of the block sub-matrix
        #pragma unroll
        for(int k=0; k<BLOCK_SIZE; ++k)
        {
          Csub += As[j][threadIdx.y][k] * Bs[j][k][threadIdx.x];
        }
        pipe.consumer_release();
      
        //Don't have to synchronize cos maxPipelineStages is greater than one therefore next
        //iteration is loading to a different buffer.
    }
    //Write the block sub-matrix to device memory;each writes 4 elements
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE*blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}
 
//Single stage memcpy async pipeline with Large copy chunk(float4)
template <int BLOCK_SIZE> __global__ void MatrixMulAsyncCopyLargeChunk(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int wA, int wB)
{
    //requires BLOCK_SIZE %4 == 0
    __shared__ alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];
    
    //Declaration of the shared memory array Bs used to store the sub-matrix of B.
    __shared__ alignas(alignof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];
  
    //Index of the first sub-matrix of A processed by the block
    int aBegin = wA*BLOCK_SIZE * blockIdx.y;
  
    //Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA -1;
  
    //step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
  
    int bBegin = BLOCK_SIZE * blockIdx.x;
  
    int bStep = BLOCK_SIZE*wB;
  
    //single stage pipeline version
    float Csub = 0.0;
  
    const int t4x = threadIdx.x*4;
    const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  
    //loop over all sub-matrices of A and B
    for(int a=aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep)
    {
      //load the matrices from device mem to shared mem.
      //A subset of threads loads a contiguous chunk of elements.
//       As[ty][tx] = A[a + wA*ty + tx];
//       Bs[ty][tx] = B[b + wB*ty + tx];
     if(t4x < BLOCK_SIZE
     {
        pipe.producer_acquire();
        cuda::memcpy_async(&As[threadIdx.y][t4x], &A[a+wA*threadIdx.y+t4x], shape4, pipe);
        cuda::memcpy_async(&Bs[threadIdx.y][t4x], &B[a+wA*threadIdx.y+t4x], shape4, pipe);
        
        pipe.producer_commit();
        pipe.consumer_wait();
     } 
     //Synchronize to make sure the matrices are loaded
     __syncthreads();
        
     //Multiply the 2 matrices together;
     //each thread computes one element of the block sub-matrix
     for(int k=0; k<BLOCK_SIZE; ++k)
     {
         Csub += As[threadIdx.y][k]*Bs[k][threadIdx.x];
     }
        
     pipe.consumer_release();
     //synchronize to make sure that the preceding computation is done before overwriting
     //the shared mem sub-matrix buffers As and Bs in the next iteration.
     __syncthreads();
    }
    //write the block sub-matrix to device mem, each thread writes 4 element
    int c = wB*BLOCK_SIZE*blockIdx.y + BLOCK_SIZE*blockIdx.x;
    C[c + wB*threadIdx.y + threadIdx.x] = Csub;
}
    
template <int BLOCK_SIZE> __global__ void MatrixMulAsyncCopyLargeChunkAWBarrier(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int wA, int wB)
{
#if __CUDA_Arch__ >= 700
#pragma diag_suppress static_var_with_dynamic_init
    //requires BLOCK_SIZE %4 == 0
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    
    //Declaration of the shared mem array As used to store the sub-matrix of A.
    __shared__ alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];
  
    //Declaration of the shared memory array Bs used to store the sub-matrix of B.
    __shared__ alignas(alingof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];
  
    if(threadIdx.x == 0)
    {
        init(&bar, blockDim.x*blockDim.y);
    }
    __syncthreads();
    
    //Index of the first sub-matrix of A processed by the block
    int aBegin = wA*BLOCK_SIZE*blockIdx.y;
    //Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;
    //step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
  
    //index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE *blockIdx.x;
    
    //step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;
  
    float Csub = 0.0;
    const int t4x = threadIdx.x*4;
  
    //loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for(int a=aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep)
    {
        //Now, one fourth of the threads load 4 elements of each matrix
        if(t4x < BLOCK_SIZE)
        {
            float4* const A4s = reinterpret_cast<float4*>(& As[threadIdx.y][t4x]);
            float4* const B4s = reinterpret_cast<float4*>(& Bs[threadIdx.y][t4x]);
            const float4* const A4 = reinterpret_cast<const float4*>(& A[a+wA*threadIdx.y+t4x]);
            const float4* const B4 = reinterpret_cast<const float4*>(& );
          
            cuda::memcpy_async(A4s, A4, sizeof(float4), bar);
            cuda::memcpy_async(B4s, B4, sizeof(float4), bar);
        }
      
        //Synchronize to make sure the matrices are loaded.
        bar.arrive_and_wait();
        //Multiply the two matrices; each thread computes 
        //one element of the block sub-matrix
#pragma unroll
        for(int k=0; k<BLOCK_SIZE; ++k)
        {
            Csub += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        }
      
        //Synchronize to make sure that the preceding computation is done 
        //before overwriting the shared mem sub-matrix buffers As and Bs.
        bar.arrive_and_wait();
    }
    //Write the block sub-matrix to device mem;
    //each thread writes 4 element
    int c = wB*BLOCK_SIZE*blockIdx.y + BLOCK_SIZE*blockIdx.x;
    C[c+wB*threadIdx.y+threadIdx.x] = Csub;
#endif
}

//Single stage memcpy async pipeline with float copy.
template <int BLOCK_SIZE> __global__ void MatrixMulAsyncCopySingleStage(
    float* C, const float *A, const float *B, int wA, int wB){
    //Declaration of the shared mem array As used to store the sub-matrix of A.
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    //Declaration of the shared mem array Bs used to store the sub-matrix of B.
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
  
    //Index of the first sub-matrix of A processed by the block
    int aBegin = wA*BLOCK_SIZE*blockIdx.y;
    //Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA -1;
    
    //step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
    
    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE*blockIdx.x;
    
    //step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE*wB;
  
    //Single stage pipeline version
    float Csub = 0.0;
  
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape1 = cuda::aligned_size_t<alignof(float)>(sizeof(float));
  
    //loop over all sub-matrices of A and B required to compute the block sub-matrix
    for(int a=aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep)
    {
      {
        //load the matrices from device mem to shared mem; each thread
        //loads one element of each matrix
        pipe.producer_acquire();
        cuda::memcpy_async(&As[threadIdx.y][threadIdx.x], &A[a+wA*threadIdx.y+threadIdx.x], shape1, pipe);
        cuda::memcpy_async(&Bs[threadIdx.y][threadIdx.x], &B[b+wB*threadIdx.y+threadIdx.x], shape1, pipe);
        pipe.producer_commit();
      }
      
      pipe.consumer_wait();
      //synchronize to make sure the matrices are loaded.
      __syncthreads();
      //Multiply the two matrices,each thread computes one element of the block sub-matrix
#pragma unroll
      for(int k=0; k<BLOCK_SIZE; ++k)
      {
          Csub += As[threadIdx.y][k]*Bs[k][threadIdx.x];
      }
      //synchronize to make sure that the preceding computation is done
      //before overwriting the shared mem sub-matrix buffers As and Bs
      __syncthreads();
    }
    //Write the block sub-matrix to device mem;
    //each thread writes 4 element.
    int c = wB*BLOCK_SIZE*blockIdx.y + BLOCK_SIZE*blockIdx.x;
    C[c + wB*threadIdx.y+threadIdx.x] = Csub;
}
        
template <int BLOCK_SIZE> __global__ void MatrixMulNaive(float* c, float* A, float* B,
                                                         int wA, int wB)
{
    //Declaration of the shared mem array As used to store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    //Index of the first sub-matrix of A processed by the block
    int aBegin = wA*BLOCK_SIZE*blockIdx.y;
    int aEnd = aBegin+wA-1;
    int aStep = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE*blockIdx.x;
    int bStep = BLOCK_SIZE*wB;
    float Csub = 0;
    for(int a=aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep)
    {
        //load the matrices from device memory to shared memory,each thread 
        //loads one element of each matrix
        As[threadIdx.y][threadIdx.x] = A[a+wA*threadIdx.y+threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[b+wB*threadIdx.y+threadIdx.x];
        //synchronize to make sure the matrices are loaded
        __syncthreads();
#pragma unroll
        for(int k=0; k<BLOCK_SIZE; ++k)
        {
            Csub += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    //write the block sub-matrix to device memory;
    //each thread writes one element
    int c = wB*BLOCK_SIZE*blockIdx.y + BLOCK_SIZE*blockIdx.x;
    C[c+wB*threadIdx.y+threadIdx.x] = Csub;
}

template <int BLOCK_SIZE> __global__ void MatrixMulNaiveLargeChunk(
    float* C, float* A, float* B, int wA, int wB
    )
{
    //Declaration of the shared memory array As used to store the sub-matrix of A
    __shared__ alignas(alignof(float4)) float As[BLOCK_SIZE][BLOCK_SIZE];
    //Declaration of the shared memory array Bs used to store the sub-matrix of B
    __shared__ alignas(alignof(float4)) float Bs[BLOCK_SIZE][BLOCK_SIZE];
    int t4x = threadIdx.x*4;
  
    //Index of the first sub-matrix of A processed by the block
    int aBegin = wA*BLOCK_SIZE*blockIdx.y;
    //Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin+wA-1;
    //step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
    
    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE*blockIdx.x;
    //step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE*wB;
    //Csub is used to store the element of the block sub-matrix
    //that is computed by the thread
    float Csub = 0;
    
    //Loop over all the sub-matrices of A and B required to compute
    //the block sub-matrix
    for(int a=aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep)
    {
        //load the matrices from device memory to shared memory.
        //One fourth of the threads load 4 elements of each matrix
        if( t4x<BLOCK_SIZE)
        {
            float4* const A4s = reinterpret_cast<float4*>(& As[threadIdx.y][t4x]);
            float4* const B4s = reinterpret_cast<float4*>(& Bs[threadIdx.y][t4x]);
            const float4* const A4 = reinterpret_cast<float4*>(& A[a+wA*threadIdx.y+t4x]);
            const float4* const B4 = reinterpret_cast<float4*>(& B[a+wA*threadIdx.y+t4x]);
            *A4s = *A4;
            *B4s = *B4;
        }
        //Synchronize to make sure the matrices are loaded
        __syncthreads();
        //Multiply the two matrices together;
        //each thread computes one element of the block sub-matrix
#pragma unroll
        for(int k=0; k<BLOCK_SIZE; ++k)
        {
          Csub += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        }
      
        //Synchronize to make sure that the preceding computation is done 
        //before loading two new sub-matrices of A and B in the next iteration.
        __syncthreads();
    }
    //write the block sub-matrix to device memory;
    //each thread writes one element
    int c=wB*BLOCK_SIZE*blockIdx.y + BLOCK_SIZE*blockIdx.x;
    C[c+wB*threadIdx.y+threadIdx.x] = Csub;
}
        
void ConstantInit(float* data, int size, float val)
{
    for(int i=0; i<size; ++i)
    {
        data[i] = val;
    }
}

int MatrixMultiply(int argc, char** argv,
                   const dim3 &dimsA,
                   const dim3 &dimsB,
                   kernels kernel_number)
{
    //Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x*dimsA.y;
    unsigned int mem_size_A = sizeof(float)* size_A;
    float* h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float)*size_B;
    float* h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
    cudaStream_t stream;
  
    //Initialize host memory
    const float valB = 2.10f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);
    
    //Allocate device memory
    float *d_A, *d_B, *d_C;
    //Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x*dimsC.y*sizeof(float);
    float* h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
    
    if(h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
  
    //Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  
    //copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
  
    //setup execution parameters
    dim3 threads(blockSize, blockSize);
    dim3 grid(dimsB.x / threads.x, dimsA.y/threads.y);
  
    //The block size is 16*18, 16 rows are consumer thread group and 
    //last 2 rows(1 wrap) is producer thread group
    dim3 threadsSharedStateKernel(blockSize, blockSize+2, 1);
    dim3 gridSharedStateKernel(dimsB.x/threadsSharedStateKernel.x, dimsA.y/threadsSharedStateKernel.x);
    
    printf("Running kernel = %d-%s\n", kernel_number, kernelNames[kernel_number]);
    //Create and start timer
    printf("Computing result using CUDA kernel...\n");
  
    //performs warmup ops using matrixMul CUDA kernel
    switch(kernel_number)
    {
      case AsyncCopyMultiStageLargeChunk:
      default:
        MatrixMulAsyncCopyMultiStageLargeChunk<blockSize><<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
      case AsyncCopyLargeChunk:
        MatrixMulAsyncCopyLargeChunk<blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
      case AsyncCopyLargeChunkAWBarrier:
        MatrixMulAsyncCopyLargeChunkAWBarrier<blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;
      case AsyncCopyMultiStageSharedState:
        MatrixMulAsyncCopyMultiStageSharedState<blockSize><<<gridSharedStateKernel, threadsSharedStateKernel, 0, stream>>>
                                                          (d_C, d_A, d_B, dimsA.x, dimsB.x);
        break;      
    }
    printf("done\n");
  
    //Execute the kernel
    int nIter = 100;
    //Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));
    
    for(int j=0; j<nIter; ++j)
    {
      switch(kernel_number)
      {
        case AsyncCopyMultiStageLargeChunk:
        default:
          MatrixMulAsyncCopyMultiStageLargeChunk<blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
          break;
        case AsyncCopyLargeChunk:
          MatrixMulAsyncCopyLargeChunk<blockSize><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
          break;
        case AsyncCopyMultiStageSharedState:
          break;
        case AsyncCopyLargeChunkAWBarrier:
          MatrixMulAsyncCopyLargeChunkAwBarrier<><<<>>>();
          break;
      }
    }
  
    //Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));
    //wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    
    //compute and print the performance
    float msecPerMatrixMul = msecTotal/nIter;
    double flopsPerMatrixMul = 2.0*static_cast<double>(dimsA.x)*
                                   static_cast<double>(dimsA.y)*
                                   static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul*1.0e-9f)/(msecPerMatrixMul/1000.0f);
    printf("Performance GFlops/s = %.2f, Time=%.3f msec, Size=%.0f Ops," \
           "WorkgroupSize=%u threads/block\n", gigaFlops, msecPerMatrixMul, flopsPerMatrixMul,
            threads.x*threads.y);
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    printf("checking computed result for correctness:");
    bool correct = true;
  
    double eps = 1.e-6; 
    for(int i=0; i<static_cast<int>(dimsC.x*dimsC.y); ++i)
    {
      double abs_err = fabs(h_C[i]-(dimsA.x*valB));
      double dot_length = dimsA.x;
      double abs_val = fabs(h_C[i]);
      double rel_err = abs_err/abs_val/dot_length;
      
      if(rel_err>eps)
      {
        printf("Error Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
        correct = false;
      }
    }
    printf("%s\n", correct ?"Result = PASS":"Result = FAIL");
  
    //clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
  
    printf("The CUDA samplese are not meant for performance measurements.");
    if(correct)
    {
      return EXIT_SUCCESS;
    } else {
      return EXIT_FAILURE;
    }
}

int main(int argc, char** argv)
{
  //This will pick the best possible CUDA capable device,
  //override the device ID based on input provided at the CML.
  int dev = findCudaDevice(argc, (const char**)argv);
  
  int matrixBlock = 32;
  dim3 dimsA(10*4*matrixBlock, 10*4*matrixBlock, 1);
  dim3 dimsB(10*4*matrixBlock, 10*4*matrixBlock, 1);
  
  //width of matrix A
  if(checkCmdLineFlag(argc, (const char**)argv, 'wA'))
  {
    dimsA.x = getCmdLineArgumentInt(argc, (const char**)argv, 'wA');
  }

  //width of matrix B
  if(checkCmdLineFlag(argc, (const char**)argv, 'wB'))
  {
    dimsB.x = getCmdLineArgumentInt(argc, (const char**)argv, 'wB');
  }
  
  if(dimsA.x != dimsB.y)
  {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }
  
  kernels selected_kernel = AsyncCopyMultiStageLargeChunk;
  if(checkCmdLineFlag(argc, (const char**)argv, "kernel"))
  {
    int kernel_number = getCmdLineArgumentInt(argc, (const char**)argv, "kernel");
    if(kernel_number < 8)
    {
      selected_kernel = (kernels)kernel_number;
    } else {
      printf("Error: kernel number should be between 0 to 6, you have entered %d\n", kernel_number);
      exit(EXIT_FAILURE);
    }
  }
  
  int major = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  if(major<7)
  {
    printf("globalToShmemAsyncCopy requires SM 7.0 or higher. Exiting...\n");
    exit(EXIT_WAIVED);
  }
  printf("MatrixA(%d,%d), MatrixB(%d, %d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
  
  int matrix_result = MatrixMultiply(argc, argv, dimsA, dimsB, selected_kernel);
  exit(matrix_result);
}

















