#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       unsigned int outputSize)
{
  __shared__ double tmp[THREADS_PER_BLOCK];
  
  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;
  
  double temp_sum = 0.0;
  for(int i=globaltid; i< inputSize; i+= gridDim.x * blockDim.x)
  {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;
  
  cg::sync(cta);
  
  cg::thread_block_title<32> tile32= cg::tiled_partition<32>(cta);
  
  double beta = temp_sum;
  double temp;
  
  for(int i=tile32.size()/2; i>0; i>>= 1)
  {
    if(tile32.thread_rank() < 1)
    {
      temp = tmp[cta.thread_rank()+i];
      beta += temp;
      tmp[cta.thread_rank()] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);
  
  if(cta.thread_rank() == 0 && blockIdx.x < outputSize)
  {
    beta = 0.0;
    for(int i=0; i<cta.size(); i+=tile32.size())
    {
      beta += tmp[i];
    }
    outputVec[blockIdx.x] = beta;
  }
}

void init_input(float *a, size_t size)
{
  for(size_t i=0; i<size; i++)
  {
    a[i] = (rand()&0xFF) / (float)RAND_MAX;
  }
}

void cudaGraphsManual(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                      double *result_d, size_t inputSize, unsigned int numOfBlocks)
{
  cudaStream_t streamForGraph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> nodeDependencies;
  cudaGraphNode_t memcpyNode, kernelNode, memsetNode;
  double result_h = 0.0;
  
  checkCudaErrors();
  cudaKernelNodeParams kernelNodeParams = {0};
  cudaMemcpy3DParams memcpyParams = {0};
  cudaMemsetParams memsetParams = {0};
  
  memcpyParams.scrArray = NULL;
  memcpyParams.srcPos = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr = make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, 
                                            inputSize, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr = make_cudaPitchedPtr(inputVec_d, sizeof(float)*inputSize, inputSize, 1);
  memcpyParams.extent = make_cudaExtent(sizeof(float)*inputSize, 1, 1);
  memcpyParams.kind = cudaMemcpyHostToDevice;
}



















