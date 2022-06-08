//This sample illustrates the usage of CUDA events for both GPU timing and overlapping CPU
//and GPU execution. Events are inserted into a stream of CUDA calls. Since CUDA stream
//calls are asynchronous, the CPU can perform computations while GPU is executing.CPU can 
//query CUDA events to determine whether GPU has completed tasks.
//system
#include <stdio.h>
//CUDA runtime
#include <cuda_runtime.h>
//project
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void increment_kernel(int* g_data, int inc_value)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx]+inc_value;
}

bool correct_output(int* data, const int n, const int x)
{
  for(int i=0; i<n; ++i)
  {
    if(data[i] != x)
    {
      printf(i, data[i], x);
      return false;
    }
  }
  
  return true;
}

int main(int argc, char* argv[])
{
  int devID;
  cudaDeviceProp deviceProps;
  
  printf("%s - Starting...\n", argv[0]);
  
  //This will pick the best possible CUDA capable device
  devID = findCudaDevice(argc, (const char **)argv);
  
  //get device name
  checkCudaErors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device %s.\n", deviceProps.name);
  
  int n=16*1024*1024;
  int bbytes = n*sizeof(int);
  int value = 26;
  
  //allocate host memory
  int *a = 0;
  checkCUdaErrors(cudamallocHost((void **)&a, nbytes));
  memset(a, 0, nbytes);
  
  //allocate device memory
  int *d_a = 0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));
  
  //set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks = dim3(n/threads.x, 1);
  
  //create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  
  StopWatchInterface *timer = NULL;
  sdkCreatetimer(&timer);
  sdkResetTimer(&timer);
  
  checkCudaErrors(cudaDeviceSynchronize());
  float gpu_time = 0.0f;
  
  //asynchronously issue work to GPU
  sdkStartTimer(&timer);
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_a, a, nbytes, cudamemcpyHostToDevice, 0);
  increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
  cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);
  sdkStopTimer(&timer);
  
  //have CPU do some work while waiting for stage 1 to finish.
  unsigned long int counter = 0;
  
  while(cudaEventQuery(stop) == cudaErrorNotReady)
  {
    counter++;
  }
  
  checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
  
  //print the cpu and gpu times
  printf("time spent executing by the GPU:%.2f\n", gpu_time);
  printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
  printf("CPU excuted %lu iterations while waiting for GPU to finish\n", counter);
  
  //check the output for correctness.
  bool bFinalResults = correct_output(a, n, value);
  
  //release resources
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFreeHost(a));
  checkCudaErrors(cudaFree(d_a));
  
  exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
