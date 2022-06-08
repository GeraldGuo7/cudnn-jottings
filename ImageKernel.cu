#include <cuda.h>
#include <helper_cuda.h>
#include <helper_image.h>

//convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	
	return ((unsigned int)(rgba.w * 255.0f)<<24 | ()()<<16);
}

static __global__ void transformKernel(unsigned int *outputData, 
																			 int width, int height, float theta,
																			 cudaTextureObject_t tex)
{
	unsigned int x= blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y= blockIdx.y*blockDim.y + threadIdx.y;
	
	float u = (float)x - (float)width /2;
	float v = (float)y - (float)height/2;
	float tu = u*cosf(theta)-v*sinf(theta);
	float tv = v*cosf(theta)+u*sinf(theta);
	tu/= (float)width;
	tv/= (float)height;
	
	//read from texture and write to global memory
	float4 pix = tex2D<float4>(tex, tu+0.5f, tv+0.5f);
	unsigned int pixelInt = rgbaFloatToInt(pix);
	outputData[y*width+x] = pixelInt;
}

static __global__ void rgbToGrayscaleKernel(unsigned int *rgbaImage,
																					 	size_t imageWidth,
																					 	size_t imageHeight)
{
	size_t gidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	uchar4 *pixArray = (uchar4 *)rgbaImage;
	
	for(int pixId = gidX; pixId<imageWidth*imageHeight; pixId+=gridDim.x*blockDim.x)
	{
		uchar4 dataA = pixArray[pixId];
		unsigned char grayscale = (unsigned char)(dataA.x*0.3 + dataA.y*0.59 +dataA.z*0.11);
		uchar4 dataB = make_uchar4(grayscale, grayscale, grayscale, 0);
		pixArray[pixId]= dataB;
	}
}

void rotateKernel(cudaTextureObject_t &texObj, const float angle, unsigned int *d_outputData,
								 	const int imageWidth, const int imageHeight, cudaStream_t stream)
{
	dim3 dimBlock(8, 8, 1);
	dim3 dimGrid(imageWidth/dimBlock.x, imageHeight/dimBlock.y, 1);
	
	transformKernel<<<dimGrid, dimBlock, 0, stream>>>(d_outputData, imageWidth, imageHeight,
																									  angle, texObj);
}





















