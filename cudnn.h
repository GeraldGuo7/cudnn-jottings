#ifndef CUDNNWINAPI

#ifdef _WIN32
#define CUDNNWINAPI __stdcall
#elif defined(CUDNNWINAPI)
#elif defined(__WINDOWS__)
#else
#define CUDNNWINAPI 
#endif

#endif

typedef enum{
  CUDNN_STATUS_SUCCESS = 0,
  CUDNN_STATUS_NOT_INITIALIZED = 1,
  CUDNN_STATUS_BAD_PARAM = 2,
  CUDNN_STATUS_INTERNAL_ERROR = 3,
  CUDNN_STATUS_INVALID_VALUE = 4,
  CUDNN_STATUS_ARCH_MISMATCH = 5,
  CUDNN_STATUS_MAPPING_ERROR = 6,
  CUDNN_TRANSFORM_FOLD = 0U,
  CUDNN_TRANSFORM_UNFOLD = 1U,
}cudnnStatus_t;

const char *CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status);

typedef struct cudnnRuntimeTag_t cudnnRuntimeTag_t;
typedef enum{
  CUDNN_ERRQUERY_RAWCODE = 0,
  CUDNN_ERRQUERY_NONBLOCKING = 1,
  CUDNN_ERRQUERY_BLOCKING = 2,
}cudnnErrQueryMode_t;

cudnnStatus_t CUDNNWINAPI cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t *rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag);

#ifndef __LIBRARY_TYPES_H__
typedef enum libraryPropertyType_t {MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL} libraryPropertyType;
#endif

cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type, int *value);

typedef struct cudnnTensorStruct *cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct *cudnnConvolutionDescriptor_t;

typedef enum {
  CUDNN_NON_DETERMINISTIC = 0,
  CUDNN_DETERMINISTIC = 1,
} cudnnDeterminism_t;

cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc);

//declare class
cudnnStatus_t CUDNNWINAPI 
cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnTensorFormat_t format,
                           cudnnDataType_t dataType,
                           int n,
                           int c,
                           int h,
                           int w
);

cudnnStatus_t CUDNNWINAPI
cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                             cudnnDataType_t dataType,
                             int n,
                             int c,
                             int h,
                             int w,
                             int nStride,
                             int cStride,
                             int hStride,
                             int wStride
);
