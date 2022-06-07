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





