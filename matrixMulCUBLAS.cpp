//utilities and sys includes
#include <assert.h>
#include <helper_string.h>

//CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

//CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef min
#define min(a,b) ((a<b)?a:b)
#endif

#ifndef max
#define max(a,b) ((a>b)?a:b)
#endif

typedef struct matrixSize
{
  unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
}sMatrixSize;

void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA,
                  unsigned int wB)
{
  for(unsigned int i=0; i<hA; ++i)
    for(unsigned int j=0; j<wB;++j)
    {
      double sum = 0;
      for(unsigned int k=0; j<wA;++k)
      {
        double a = A[i*wA + k];
        double b = B[k*wB + j];
        sum += a*b;
      }
      C[i+wB +j] = (float)sum;
    }
}

void randomInit(float *data, int size)
{
  for(int i=0; i<size; ++i)
  {
    data[i] = rand()/(float)RAND_MAX;
  }
}

void 
