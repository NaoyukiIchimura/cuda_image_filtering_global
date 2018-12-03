////
//// cuda_image_filtering_global.cu: an example program for image filtering using CUDA, global memory version
////

///
/// The standard include files
///
#include <iostream>
#include <string>

#include <cmath>

///
/// The include files for CUDA
///
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

///
/// The include files for image filtering
///
#include "path_handler.h"
#include "image_rw_cuda.h"
#include "padding.h"
#include "postprocessing.h"
#include "get_micro_second.h"

///
/// An inline function of the ceilling function for unsigned int variables
///
inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b ) { return ( a%b != 0 ) ? (a/b+1):(a/b); }

///
/// The kernel function for image filtering using only global memory
/// Note that passing references cannot be used.
///
template <typename T>
__global__ void imageFilteringKernel( const T *d_f, const unsigned int paddedW, const unsigned int paddedH,
                                      const T *d_g, const int S,
                                      T *d_h, const unsigned int W, const unsigned int H )
{

    // Set the padding size and filter size
    unsigned int paddingSize = S;
    unsigned int filterSize = 2 * S + 1;

    // Set the pixel coordinate
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

    // The multiply-add operation for the pixel coordinate ( j, i )
    if( j >= paddingSize && j < paddedW - paddingSize && i >= paddingSize && i < paddedH - paddingSize ) {
        unsigned int oPixelPos = ( i - paddingSize ) * W + ( j - paddingSize );
        d_h[oPixelPos] = 0.0;
        for( int k = -S; k <= S; k++ ) {
            for( int l = -S; l <= S; l++ ) {
                unsigned int iPixelPos = ( i + k ) * paddedW + ( j + l );
                unsigned int coefPos = ( k + S ) * filterSize + ( l + S );
                d_h[oPixelPos] += d_f[iPixelPos] * d_g[coefPos];
            }
        }
    }

}

///
/// The function for image filtering performed on a CPU
///
template <typename T>
int imageFiltering( const T *h_f, const unsigned int &paddedW, const unsigned int &paddedH,
                    const T *h_g, const int &S,
                    T *h_h, const unsigned int &W, const unsigned int &H )
{

    // Set the padding size and filter size
    unsigned int paddingSize = S;
    unsigned int filterSize = 2 * S + 1;

    // The loops for the pixel coordinates
    for( unsigned int i = paddingSize; i < paddedH - paddingSize; i++ ) {
        for( unsigned int j = paddingSize; j < paddedW - paddingSize; j++ ) {

            // The multiply-add operation for the pixel coordinate ( j, i )
            unsigned int oPixelPos = ( i - paddingSize ) * W + ( j - paddingSize );
            h_h[oPixelPos] = 0.0;
            for( int k = -S; k <=S; k++ ) {
                for( int l = -S; l <= S; l++ ) {
                    unsigned int iPixelPos = ( i + k ) * paddedW + ( j + l );
                    unsigned int coefPos = ( k + S ) * filterSize + ( l + S );
                    h_h[oPixelPos] += h_f[iPixelPos] * h_g[coefPos];
                }
            }

        }
    }

    return 0;

}       

///
/// Comopute the mean squared error between two images
///
template <typename T>
T calMSE( const T *image1, const T *image2, const unsigned int &iWidth, const unsigned int &iHeight )
{

    T mse = 0.0;
    for( unsigned int i = 0; i < iHeight; i++ ) {
	for( unsigned int j = 0; j < iWidth; j++ ) {
	    unsigned int pixelPos = i * iWidth + j;
	    mse += ( image1[pixelPos] - image2[pixelPos] ) * ( image1[pixelPos] - image2[pixelPos] );
	}
    }
    mse = sqrt( mse );

    return mse;

}

///
/// The main function
///
int main( int argc, char *argv[] )
{

    //----------------------------------------------------------------------

    //
    // Declare the variables for measuring elapsed time
    //
    double sTime;
    double eTime;

    //----------------------------------------------------------------------

    //
    // Input file paths
    //
    std::string inputImageFilePath;
    std::string filterDataFilePath;
    std::string outputImageFilePrefix;
    if( argc <= 1 ) {
	std::cerr << "Input image file path: ";
	std::cin >> inputImageFilePath;
	std::cerr << "Filter data file path: ";
	std::cin >> filterDataFilePath;
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePrefix;
    } else if( argc <= 2 ) {
	inputImageFilePath = argv[1];
	std::cerr << "Filter data file path: ";
	std::cin >> filterDataFilePath;
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePrefix;
    } else if( argc <= 3 ) {
	inputImageFilePath = argv[1];
	filterDataFilePath = argv[2];
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePrefix;
    } else {
	inputImageFilePath = argv[1];
	filterDataFilePath = argv[2];
	outputImageFilePrefix = argv[3];
    }

    //----------------------------------------------------------------------

    //
    // Set the prefix and extension of the input image file
    //
    std::string imageFileDir;
    std::string imageFileName;
    getDirFileName( inputImageFilePath, &imageFileDir, &imageFileName );

    std::string imageFilePrefix;
    std::string imageFileExt;
    getPrefixExtension( imageFileName, &imageFilePrefix, &imageFileExt );

    //----------------------------------------------------------------------

    //
    // Read the intput image in pageable memory on a host
    // Page-locked memory (write-combining memory) is not used, because padding is performed on a host 
    //
    hsaImage<float> h_inputImage;
    if( imageFileExt == "tif" ) { // TIFF
      h_inputImage.tiffGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.tiffReadImage( inputImageFilePath );
    } else if( imageFileExt == "jpg" ) { // JPEG
      h_inputImage.jpegGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.jpegReadImage( inputImageFilePath );
    } else if( imageFileExt == "png" ) { // PNG
      h_inputImage.pngGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.pngReadImage( inputImageFilePath );
    }

    //
    // Show the size of the input image
    //
    std::cout << "The size of the input image: ("
	      << h_inputImage.getImageWidth()
	      << ", "
	      << h_inputImage.getImageHeight() 
	      << ")"
	      << std::endl;

    //----------------------------------------------------------------------

    //
    // Prepare a Y component image
    //
    float *h_image;
    unsigned int iWidth = h_inputImage.getImageWidth();
    unsigned int iHeight = h_inputImage.getImageHeight();
    try {
	h_image = new float[ iWidth * iHeight ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_image: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    // Compute Y component
    for( unsigned int i = 0; i < h_inputImage.getImageHeight(); i++ ) {
	for( unsigned int j = 0; j < h_inputImage.getImageWidth(); j++ ) {
	    unsigned int pixelPos = i * h_inputImage.getImageWidth() + j;
	    h_image[pixelPos] = 0.2126 * h_inputImage.getImagePtr( 0 )[pixelPos] +
		0.7152 * h_inputImage.getImagePtr( 1 )[pixelPos] +
		0.0722 * h_inputImage.getImagePtr( 2 )[pixelPos];
	}
    }

    //----------------------------------------------------------------------

    //
    // Read the filter data file
    //
    std::ifstream fin;
    fin.open( filterDataFilePath.c_str() );
    if( !fin ) {
	std::cerr << "Could not open the filter data file: "
		  << filterDataFilePath
		  << std::endl;
	exit(1);
    }

    // Read the size of the filter
    unsigned int filterSize;
    fin >> filterSize;

    // Read the filter kernel
    float *h_filterKernel;
    try {
	h_filterKernel = new float[ filterSize * filterSize ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_filterKernel: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i = 0; i < filterSize; i++ )
	for( unsigned int j = 0; j < filterSize; j++ )
	    fin >> h_filterKernel[ i * filterSize + j ];

    std::cout << "*** Filter coefficients ***" << std::endl;  
    for( unsigned int i = 0; i < filterSize; i++ ) {
	for( unsigned int j = 0; j < filterSize; j++ )
	    std::cout << h_filterKernel[ i * filterSize + j ] << " ";
	std::cout << std::endl;
    }

    //----------------------------------------------------------------------

    //
    // Perform padding for the image
    //
    int hFilterSize = filterSize / 2;
    unsigned int paddedIWidth = iWidth + 2 * hFilterSize;
    unsigned int paddedIHeight = iHeight + 2 * hFilterSize;
    float *h_paddedImage;
    try {
	h_paddedImage = new float[ paddedIWidth * paddedIHeight ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_paddedImage: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    replicationPadding( h_image, iWidth, iHeight,
			hFilterSize,
			h_paddedImage, paddedIWidth, paddedIHeight );
    
    //----------------------------------------------------------------------

    //
    // Perform image filtering by a GPU 
    //

    // Transfer the padded image to a device 
    float *d_paddedImage;
    unsigned int paddedImageSizeByte = paddedIWidth * paddedIHeight * sizeof(float);
    checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&d_paddedImage), paddedImageSizeByte ) );
    sTime = getMicroSecond();
    checkCudaErrors( cudaMemcpy( d_paddedImage, h_paddedImage, paddedImageSizeByte, cudaMemcpyHostToDevice ) );
    eTime = getMicroSecond();
    double dataTransferTime = eTime - sTime;

    // Transfer the filter to a device
    float *d_filterKernel;
    unsigned int filterKernelSizeByte = filterSize * filterSize * sizeof(float);
    checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&d_filterKernel), filterKernelSizeByte ) );
    sTime = getMicroSecond();
    checkCudaErrors( cudaMemcpy( d_filterKernel, h_filterKernel, filterKernelSizeByte, cudaMemcpyHostToDevice ) );
    eTime = getMicroSecond();
    dataTransferTime += ( eTime - sTime );

    // Set the execution configuration
    const unsigned int blockW = 16;
    const unsigned int blockH = 16;
    const dim3 grid( iDivUp( iWidth, blockW ), iDivUp( iHeight, blockH ) );
    const dim3 threadBlock( blockW, blockH );
    
    // call the kernel function for image filtering
    float *d_filteringResult;
    unsigned int imageSizeByte = iWidth * iHeight * sizeof(float);
    checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&d_filteringResult), imageSizeByte ) );

    sTime = getMicroSecond();
    checkCudaErrors( cudaDeviceSynchronize() );
    imageFilteringKernel<<<grid,threadBlock>>>( d_paddedImage, paddedIWidth, paddedIHeight,
						d_filterKernel, hFilterSize,
						d_filteringResult, iWidth, iHeight );
    checkCudaErrors( cudaDeviceSynchronize() );
    eTime = getMicroSecond();
    double filteringTimeGPU = eTime - sTime;

    // Back-transfer the filtering result to a host
    float *h_filteringResultGPU;
    try {
	h_filteringResultGPU =new float[ iWidth * iHeight ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_filteringResultGPU: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    sTime = getMicroSecond();
    checkCudaErrors( cudaMemcpy( h_filteringResultGPU, d_filteringResult, imageSizeByte, cudaMemcpyDeviceToHost ) );
    eTime = getMicroSecond();
    dataTransferTime += ( eTime - sTime );

    //----------------------------------------------------------------------

    //
    // Perform image filtering by a CPU
    //
    float *h_filteringResultCPU;
    try {
	h_filteringResultCPU =new float[ iWidth * iHeight ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_filteringResultCPU: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    sTime = getMicroSecond();
    imageFiltering( h_paddedImage, paddedIWidth, paddedIHeight,
		    h_filterKernel, hFilterSize,
		    h_filteringResultCPU, iWidth, iHeight );
    eTime = getMicroSecond();
    double filteringTimeCPU = eTime - sTime;

    //----------------------------------------------------------------------

    //
    // Compare the filtering results by a GPU and a CPU
    //
    float mse = calMSE( h_filteringResultGPU, h_filteringResultCPU, iWidth, iHeight );

    std::cout << "MSE: " << mse << std::endl;

    //----------------------------------------------------------------------

    //
    // Show the compution time
    //
    std::cout << "The time for data transfer: " << dataTransferTime * 1e3 << "[ms]" <<std::endl;
    std::cout << "The time for filtering by GPU: " << filteringTimeGPU * 1e3 << "[ms]" << std::endl;
    std::cout << "The time for filtering by CPU: " << filteringTimeCPU * 1e3 << "[ms]" << std::endl;
    std::cout << "Filering: the GPU is " << filteringTimeCPU / filteringTimeGPU << "X faster than the CPU." << std::endl;
    std::cout << "The overall speed-up is " << filteringTimeCPU / ( dataTransferTime + filteringTimeGPU ) << "X." << std::endl;

    //----------------------------------------------------------------------

    //
    // Save the fitlering results
    //
    hsaImage<float> filteringResultImage;
    filteringResultImage.allocImage( iWidth, iHeight, PAGEABLE_MEMORY );

    // Set the number of channels
    const unsigned int RGB = 3;

    // The GPU result
    for( unsigned int i = 0; i < iHeight; i++ ) {
	for( unsigned int j = 0; j < iWidth; j++ ) {
	    unsigned int pixelPos = i * iWidth + j;
	    for( unsigned int k = 0; k < RGB; k++ )
		filteringResultImage.getImagePtr( k )[pixelPos] = h_filteringResultGPU[pixelPos];
	}
    }
    takeImageAbsoluteValueCPU( &filteringResultImage, RGB );
    normalizeImageCPU( &filteringResultImage, RGB );
    adjustImageLevelCPU( &filteringResultImage, RGB, static_cast<float>(255) );
    std::string filteringResultGPUFileName = outputImageFilePrefix + "_GPU.png";
    filteringResultImage.pngSaveImage( filteringResultGPUFileName, RGB_DATA );

    // The CPU result
    for( unsigned int i = 0; i < iHeight; i++ ) {
	for( unsigned int j = 0; j < iWidth; j++ ) {
	    unsigned int pixelPos = i * iWidth + j;
	    for( unsigned int k = 0; k < RGB; k++ )
		filteringResultImage.getImagePtr( k )[pixelPos] = h_filteringResultCPU[pixelPos];
	}
    }
    takeImageAbsoluteValueCPU( &filteringResultImage, RGB );
    normalizeImageCPU( &filteringResultImage, RGB );
    adjustImageLevelCPU( &filteringResultImage, RGB, static_cast<float>(255) );
    std::string filteringResultCPUFileName = outputImageFilePrefix + "_CPU.png";
    filteringResultImage.pngSaveImage( filteringResultCPUFileName, RGB_DATA );

    //----------------------------------------------------------------------

    //
    // Delete the memory spaces
    //
    filteringResultImage.freeImage();

    delete [] h_filteringResultCPU;
    h_filteringResultCPU = 0;

    delete [] h_filteringResultGPU;
    h_filteringResultGPU = 0;

    checkCudaErrors( cudaFree( d_filteringResult ) );
    d_filteringResult = 0;

    checkCudaErrors( cudaFree( d_filterKernel ) );
    d_filterKernel = 0;

    checkCudaErrors( cudaFree( d_paddedImage ) );
    d_paddedImage = 0;

    delete [] h_paddedImage;
    h_paddedImage = 0;

    delete [] h_image;
    h_image = 0;

    delete [] h_filterKernel;
    h_filterKernel = 0;

    h_inputImage.freeImage();

    //----------------------------------------------------------------------

    return 0;
    
}
