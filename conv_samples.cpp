template <typename T_ELEM, typename T_MATH>
static void 
conv_cpu_ref(const T_ELEM* inputData,
             const T_ELEM* filterData,
             T_ELEM* outputData,
             float alpha,
             float beta,
             int resizeFactor,
             cudnnTensorFormat_t filterFormat,
             const int* inDims,
             const int* filDims,
             const int* outDims,
             const int* inStride,
             const int* outStride,
             const int* stride,
             const int* pad,
             const int* dilation,
             int nbDims)
{
  int imDims = nbDims - 2;
  int filStride[8] = {0};
  generateStrides(filDims, filStride, nbDims, FilterFormat);
  
  bool isConv = true;
  //number of pixels in output
  int nPixelsOut = 1;
  for(int i=2; i<nbDims; ++i)
  {
    nPixelsOut *= outDims[i];
  }
  //number of pixels in filter
  int nPixelsFil = 1;
  for(int i=2; i<nbDims; ++i)
  {
    nPixelsOut *= outDims[i];
  }

  //Used to store coordinates
  int filIds[8] = {0};
  int outIds[8] = {0};
  int inIds[8] = {0};
  int tmpIds[8] = {0};
  
  //For each image in the output
  for(int ni = 0; ni<outDims[0]; ni++)
  {
    for(int ki_outer = 0; ki_outer < outDims[1]/resizeFactor; ki_outer++)
    {
      int outputOffset = ni * outStride[0]/resizeFactor + ki_outer*outerStride[1];
      //For every pixel in this output image's feature layer
      for(int outId=0; outId < nPixelsOut; ++outId)
      {
        lin2dim(outId, outIds, outDims+2, imDims); //Skip n and k dimensions
        //Now we get the coordinates in input space of the "top left"corner
        //of the filter; multiple by stride and remove pad
        for(int d=0; d<imDims; ++d)
        {
          inIds[d] = outIds[d]*stride[d]-pad[d];
        }
        
        //For each inner feature layer of the output image
        for(int ki_inner = 0; ki_inner<resizeFactor; ++ki_inner)
        {
          T_MATH tmp = 0;
          //For each outer feature layer of the input image and filter
          for(int ci=0; ci<inDims[1]/resizeFactor; ++ci)
          {
            int inputOffset = ni * inStride[0]/resizeFactor + ci * inStride[1];
            int filterOffset = 
              (ki_outer * resizeFactor + ki_inner)*filStride[0]/resizeFactor + ci*filStride[1];
            //Now for every pixel in the filter
            for(int filId =0; filId<nPixelsFil; ++filId)
            {
              lin2dim(filId, filIds, filDims+2, imDims);
              //Get the position of the pixel
              //Compute the corresponding output pixel and check whether we are in the padding area on the fly too
              //(not that for convolution, we flip the image patch;
              //equivalent to flipping the filter patch).
              bool inside = true;
              for(int d=0; d<imDims&&inside; ++d)
              {
                if(isConv)
                {
                  tmpIds[d] = inIds[d] + dilation[d]*(filDims[2+d]-1-filIds[d]);
                } else {
                  tmpIds[d] = inIds[d] + dilation[d]*filIds[d];
                }
                inside &= (tmpIds[d]>=0 && tmpIds[d]<inDims[2+d]);
              }
              if(inside)
              {
                int actualTmpId = inputOffset + dim2lin(tmpIds, (inStride)+2, imDims);
                //int actualFilId = filterOffset + filId;
                int actualFilId = filterOffset + dim2lin(filIds, (filStride)+2, imDims);
                
                //For each inner feature layer of the input image and filter
                for(int i=0; i<resizeFactor; ++i)
                {
                  T_ELEM fval = filterData[actualFilId*resizeFactor+i];
                  T_ELEM ival = inputData[actualTmpId*resizeFactor+i];
                  tmp = doFma(fval, ival, tmp);
                }
              }
            }
          }
          
          //Store final result in proper position in output image
          int actualOutId = outputOffset + dim2lin(outIds, (outStride)+2, imDims);
          doEpilog(outputData, actualOutId * resizeFactor + ki_inner, alpha*tmp, beta);
        }
      }
    }
  } 
}
