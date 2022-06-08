#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <nvrtc_helper.h>
#include <helper_functions.h>

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(int argc, char** argv)
{
  printf("CUDA clock sample\n");
  
  typedef long clock_t;
  clock_t timer[NUM_BLOCKS*2];
  
  float input[NUM_THREADS*2];
  
  for(int i=0; i<NUM_THREADS*2;++i)
  {
    input[i] = (float)i;
  }
  
  char *cubin, *kernel_file;
  size_t cubinSize;
  
  kernel_file = sdkFindFilePath("clock_kernel.cu", argv[0]);
  compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubinSize, 0);
  
  CUmodule module = loadCUBIN(cubin, argc, argv);
  CUfunction kernel_addr;
  
  checkCUdaErrors(cuModuleGetFunction(&kernel_addr, module, "timeReduction"));
  
  dim3 cudaBlockSize(NUM_THREADS, 1, 1);
  dim3 cudaGridSize(NUM_BLOCKS, 1, 1);
  
  CUdevicePtr dinput, doutput, dtimer;
  checkCudaErrors(cuMemAlloc(&dinput, sizeof(float) * NUM_THREADS*2));
  checkCudaErrors(cuMemAlloc(&doutput, sizeof(float)*NUM_BLOCKS));
  checkCudaErrors(cuMemcpyHtoD(dinput, input, sizeof(float)*NUM_THREADS*2));
  
  void *arr[] = { (void*)&dinput, (void*)&doutput, (void*)&dtimer};
  checkCudaErrors(cuMemFree(dinput));
  checkCudaErrors(cuMemFree(doutput));
  
  long double avgElapsedClocks = 0;
  for(int i=0; i<NUM_BLOCKS; ++i)
  {
    avgElapsedClocks += (long double) (timer[i+NUM_BLOCKS]-timer[i]);
  }
  
  avgElapsedClocks = avgElapsedClocks/NUM_BLOCKS;
  printf("Average clocks/block = %Lf.\n", avgElapsedClocks);
  
  return EXIT_SUCCESS;
}
