#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <helper_cuda.h>

namespace cg= cooperative_groups;

void initOddEvenArr(int* intputArr, unsigned int size)
{
  for(unsigned int i=0; i<size; ++i)
  {
    inputArr[i] = rand()%50;
  }
}

//CUDA kernel device code creates cooperative groups and performs odd/even counting & summation.
__global__ void oddEveCountAndSumCG(int* inputArr, int* numOfOdds, int* sumOfOddAndEvents,
                                    unsigned int size)
{
  cg::thread_block cta = cg::thread_block();
  cg::grid_group grid = cg::this_grid();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  
  for(int i=grid.thread_rnak(); i<size; i+=grid.size())
  {
    int elem = inputArr[i];
    auto subTile = cg::binary_partition(tile32, elem & 1);
    if(elem & 1)//Odd numbers group
    {
      int oddGroupSum = cg::reduce(subTile, elem, cg::plus<int>());
      
      if(subTile.thread_rank()==0)
      {
        //Add number of odds present in this group of Odds.
        atomicAdd(numOfOdds, subTile.size());
        //Add local reduction of odds present in this group of Odds.
        atomicAdd(&sumOfOddAndEvents[0], oddGroupSum);
      }
    } else {
      //Even numbers group
      int evenGroupSum = cg::reduce(subTile, elem, cg::plus<int>());
      if(subTile.thread_rank()==0)
      {
        //Add local reduction of even present in this group of events.
        atomicAdd(&sumOfOddAndEvents[i], evenGroupSum);
      }
    }
    //reconverge warp so for next loop iteration we ensure convergence of
    //above diverged threads to perform coalesced loads of inputArr.
    cg::sync(tile32);
  }
}

int main(int argc,const char** argv)
{
  int deviceId = findCudaDevice(argc, argv);
  int *h_inputArr, *d_inputArr;
  int *h_numOfOdds, *d_numOfOdds;
  int *h_sumOfOddEvenElems, *d_sumOfOddEvenElems;
  unsigned int arrSize = 1024*100;
  
  checkCudaErrors(cudaMallocHost(&h_inputArr, sizeof(int)*arrSize));
  checkCudaErrors(cudaMallocHost(&h_numOfOdds, sizeof(int)));
  checkCudaErrors(cudaMallocHost(&h_sumOfOddEvenElems, sizeof(int)*2));
  initOddEvenArr(h_inputArr, arrSize);
  
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaMalloc(&d_inputArr, sizeof(int)*arrSize));
  checkCudaErrors(cudaMalloc(&d_numOfOdds, sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_sumOfOddEvenElems, sizeof(int)*2));
  
  checkCudaErrors(cudaMemcpyAsync(d_inputArr, h_inputArr, sizeof(int)*arrSize, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemsetAsync(d_numOfOdds, 0, sizeof(int), stream));
  checkCudaErrors(cudaMemsetAsync(d_sumOfOddEvenElems, 0, 2*sizeof(int), stream));
  
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&blocksPerGrid, &threadsPerBlock, oddEvenCountAndSumCG, 0, 0));
  
  printf("\nLaunching %d blocks with %d threads...\n\n", blocksPerGrid, threadsPerBlock);
  
  oddEvenCountAndSumCG<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_inputArr, d_numOfOdds, d_sumOfOddEvenElems, arrSize);
  
  checkCudaErrors(cudamemcpyAsync(h_numOfOdds, d_numOfOdds, sizeof(int), cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
  
  printf("\n...Done.\n");
  
  checkCudaErrors(cudaFreeHost(h_inputArr));
  checkCudaErrors(cudaFreeHost(h_numOfOdds));
  checkCudaErrors(cudaFreeHost(h_sumOfOddEvenElems));
  checkCudaErrors(cudaFree(d_inputArr));
  checkCudaErrors(cudaFree(d_numOfOdds));
  checkCudaErrors(cudaFree(d_sumOfOddEvenElems));
  
  return EXIT_SUCCESS:
}
