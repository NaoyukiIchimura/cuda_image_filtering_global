////
//// image_rw_cuda.cpp: the member functions for reading/saving images
////

///
/// The standard include files
///
#include <iostream>
#include <string>
#include <new>
#include <fstream>

#include <cstdlib>
#include <cstring>

///
/// The include files for CUDA
///
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

///
/// The include file for the dsaImage and hsaImage classes
///
#include "image_rw_cuda.h"

////
//// The member functions of the image class for a device
////

///
/// The default constructor
///
template <typename T>
dsaImage<T>::dsaImage() : mWidth(0), mHeight(0), mYComp(0)
{

    //
    // Initialize the arrays
    //
    for( unsigned int i = 0; i < RGB; i++ ) {
	mData[i] = 0;
	mConvData[i] = 0;
    }

}

///
/// The destructor
///
template <typename T>
dsaImage<T>::~dsaImage()
{

}

///
/// Allocate an image 
///
template <typename T>
int dsaImage<T>::allocImage()
{

    //
    // Check the size of the image
    //
    if( mWidth <= 0 || mHeight <= 0 ) {
	std::cerr << "Image size must be positive: "
		  << "(" << mWidth << ", " << mHeight << ")"
		  << std::endl;
	exit(1);
    }
    
    //
    // Allocate the memory space for the image on a device
    //
    const unsigned int imageSizeByte = mWidth * mHeight * sizeof(T);
    for( unsigned int i = 0; i < RGB; i++ ) {
	checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&(mData[i])), imageSizeByte ) );
	checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&(mConvData[i])), imageSizeByte ) );
    }
    checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&(mYComp)), imageSizeByte ) );

    return 0;

}

///
/// Allocate an image with its size
///
template <typename T>
int dsaImage<T>::allocImage( const unsigned int &width, const unsigned int &height )
{

    //
    // Check the size of the image
    //
    if( width <= 0 || height <= 0 ) {
	std::cerr << "Image size must be positive: "
		  << "(" << width << ", " << height << ")"
		  << std::endl;
	exit(1);
    }
    
    //
    // Set the image size
    //
    mWidth = width;
    mHeight = height;

    //
    // Allocate the memory space for the image on a device
    //
    const unsigned int imageSizeByte = mWidth * mHeight * sizeof(T);
    for( unsigned int i = 0; i < RGB; i++ ) {
	checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&(mData[i])), imageSizeByte ) );
	checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&(mConvData[i])), imageSizeByte ) );
    }
    checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&(mYComp)), imageSizeByte ) );

    return 0;

}

///
/// Free the image
///
template <typename T>
int dsaImage<T>::freeImage()
{

    for( unsigned int i = 0; i < RGB; i++ ) {
	if( mData[i] ) {
	    checkCudaErrors( cudaFree(mData[i]) );
	    mData[i] = 0;
	}
	if( mConvData[i] ) {
	    checkCudaErrors( cudaFree(mConvData[i]) );
	    mConvData[i] = 0;
	}
    }

    if( mYComp ) {
	checkCudaErrors( cudaFree(mYComp) );
	mYComp = 0;
    }

    return 0;

}

//
// The functions for image transfer between a host and a device
// These are friend functions of the hsaImage class
//
template <typename T>
int dsaImage<T>::transferImage( const hsaImage<T> &h_image )
{

    //
    // Check the size of the image on a host
    //
    if( h_image.mWidth <= 0 || h_image.mHeight <= 0 ) {
	std::cerr << "The size of source image must be positive: "
		  << "(" << h_image.mWidth << ", " << h_image.mHeight << ")"
		  << std::endl;
	exit(1);
    }

    //
    // Set the image size by byte
    //
    const unsigned int imageSizeByte = h_image.mWidth * h_image.mHeight * sizeof(T);

    //
    // Copy the image even if the sizes of the images are different
    //
    if( mWidth != h_image.mWidth || mHeight != h_image.mHeight ) {

	freeImage();

	mWidth = h_image.mWidth;
	mHeight = h_image.mHeight;

	allocImage();

    }

    //
    // Transfer the source image
    //
     for( unsigned int i = 0; i < RGB; i++ )
	checkCudaErrors( cudaMemcpy( mData[i], h_image.mData[i], imageSizeByte, cudaMemcpyHostToDevice ) );

    return 0;

}

template <typename T>
int dsaImage<T>::backTransferImage( hsaImage<T> *h_image, const imagePointerNo &pointerNo ) const
{

    //
    // Check the size of the image on a device
    //
    if( mWidth <= 0 || mHeight <= 0 ) {
	std::cerr << "The size of device image must be positive: " 
		  << "(" << mWidth << ", " << mHeight << ")" 
		  << std::endl;
	exit(1);
    }
    if( mWidth != h_image->mWidth || mHeight != h_image->mHeight ) {
	std::cerr << "The sizes of the images must be the same: " 
		  << "(" << h_image->mWidth << ", " << h_image->mHeight << "), "
		  << "(" << mWidth << ", " << mHeight << ")" 
		  << std::endl;
	exit(1);
    }
    //
    // Transfer the device image
    //
    const unsigned int imageSizeByte = mWidth * mHeight * sizeof(T);
    switch( pointerNo ) {

      case RGB_DATA:
	for( unsigned int i = 0; i < RGB; i++ )
	    checkCudaErrors( cudaMemcpy( h_image->mData[i], mData[i], imageSizeByte, cudaMemcpyDeviceToHost ) );
	break;

      case Y_COMPONENT:
	checkCudaErrors( cudaMemcpy( h_image->mYComp, mYComp, imageSizeByte, cudaMemcpyDeviceToHost ) );
	break;

      case CONVERTED_DATA:
	for( unsigned int i = 0; i < RGB; i++ )
	    checkCudaErrors( cudaMemcpy( h_image->mConvData[i], mConvData[i], imageSizeByte, cudaMemcpyDeviceToHost ) );
	break;

      default: // Back-transfer the original image
	for( unsigned int i = 0; i < RGB; i++ )
	    checkCudaErrors( cudaMemcpy( h_image->mData[i], mData[i], imageSizeByte, cudaMemcpyDeviceToHost ) );
	break;
	
    }

    return 0;

}

////
//// The member functions of the image class for a host
////

///
/// The default constructor
///
template <typename T>
hsaImage<T>::hsaImage() : mWidth(0), mHeight(0), mYComp(0)
{
    
    for( unsigned int i = 0; i < RGB; i++ ) {
	mData[i] = 0;
	mConvData[i] = 0;
    }

}

///
/// The destructor
///
template <typename T>
hsaImage<T>::~hsaImage()
{

}

///
/// Allocate an image
///
template <typename T>
int hsaImage<T>::allocImage( const hostMemoryType &hostMemoryType )
{

    //
    // Check the size of the image
    //
    if( mWidth <= 0 || mHeight <= 0 ) {
	std::cerr << "Image size must be positive: "
		  << "(" << mWidth << ", " << mHeight << ")"
		  << std::endl;
	exit(1);
    }

    //
    // Set the type of host memory
    //
    mHostMemoryType = hostMemoryType;

    //
    // Allocate the memory space for the image 
    //
    if( mHostMemoryType ) {
	const unsigned int imageSizeByte = mWidth * mHeight * sizeof(T);
	for( unsigned int i = 0; i < RGB; i++ ) {
	    checkCudaErrors( cudaHostAlloc( reinterpret_cast<void **>(&(mData[i])), imageSizeByte, cudaHostAllocWriteCombined ) );
	    checkCudaErrors( cudaHostAlloc( reinterpret_cast<void **>(&(mConvData[i])), imageSizeByte, cudaHostAllocWriteCombined ) );
	}
	checkCudaErrors( cudaHostAlloc( reinterpret_cast<void **>(&(mYComp)), imageSizeByte, cudaHostAllocWriteCombined ) );
    } else {
	const unsigned int imageSize = mWidth * mHeight;
	for( unsigned int i = 0; i < RGB; i++ ) {
	    try {
		mData[i] = new T[imageSize];
	    } catch( std::bad_alloc & ) {
		std::cerr << "Could not allocate the memory space for mData: " 
			  << __FILE__ << " : " << __LINE__
			  << std::endl;
		exit(1);
	    }
	    try {
		mConvData[i] = new T[imageSize];
	    } catch( std::bad_alloc & ) {
		std::cerr << "Could not allocate the memory space for mConvData: " 
			  << __FILE__ << " : " << __LINE__
			  << std::endl;
		exit(1);
	    }
	}
	try {
	    mYComp = new T[imageSize];
	} catch( std::bad_alloc & ) {
	    std::cerr << "Could not allocate the memory space for mYComp: "
		      << __FILE__ << " : " << __LINE__ 
		      << std::endl;
	    exit(1);
	}
	// Clear the memory space
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		mYComp[pixelPos] = 0;
		for( unsigned int k = 0; k < RGB; k++ ) {
		    mData[k][pixelPos] = 0;
		    mConvData[k][pixelPos] = 0;
		}
	    }
	}
    }

    return 0;	
	
}

///
/// Allocate an image with its size
///
template <typename T>
int hsaImage<T>::allocImage( const unsigned int &width, const unsigned int &height, const hostMemoryType &hostMemoryType )
{

    //
    // Check the size of the image
    //
    if( width <= 0 || height <= 0 ) {
	std::cerr << "Image size must be positive: "
		  << "(" << width << ", " << height << ")"
		  << std::endl;
	exit(1);
    }

    //
    // Set image size
    //
    mWidth = width;
    mHeight = height;

    //
    // Set the type of host memory
    //
    mHostMemoryType = hostMemoryType;

    //
    // Allocate memory space for the image 
    //
    if( mHostMemoryType ) {
	const unsigned int imageSizeByte = mWidth * mHeight * sizeof(T);
	for( unsigned int i = 0; i < RGB; i++ ) {
	    checkCudaErrors( cudaHostAlloc( reinterpret_cast<void **>(&(mData[i])), imageSizeByte, cudaHostAllocWriteCombined ) ); 
	    checkCudaErrors( cudaHostAlloc( reinterpret_cast<void **>(&(mConvData[i])), imageSizeByte, cudaHostAllocWriteCombined ) ); 
	}
	checkCudaErrors( cudaHostAlloc( reinterpret_cast<void **>(&(mYComp)), imageSizeByte, cudaHostAllocWriteCombined ) );
    } else {
	const unsigned int imageSize = mWidth * mHeight;
	for( unsigned int i = 0; i < RGB; i++ ) {
	    try {
		mData[i] = new T[imageSize];
	    } catch( std::bad_alloc & ) {
		std::cerr << "Could not allocate the memory space for mData: " 
			  << __FILE__ << " : " << __LINE__
			  << std::endl;
		exit(1);
	    }
	    try {
		mConvData[i] = new T[imageSize];
	    } catch( std::bad_alloc & ) {
		std::cerr << "Could not allocate the memory space for mConvData: " 
			  << __FILE__ << " : " << __LINE__
			  << std::endl;
		exit(1);
	    }
	}
	try {
	    mYComp = new T[imageSize];
	} catch( std::bad_alloc & ) {
	    std::cerr << "Could not allocate the memory space for mYComp: "
		      << __FILE__ << " : " << __LINE__
		      << std::endl;
	    exit(1);
	}
	// Clear the memory space
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		mYComp[pixelPos] = 0;
		for( unsigned int k = 0; k < RGB; k++ ) {
		    mData[k][pixelPos] = 0;
		    mConvData[k][pixelPos] = 0;
		}
	    }
	}
    }
    
    return 0;	
	
}

///
/// Free the image
///
template <typename T>
int hsaImage<T>::freeImage()
{

    if( mHostMemoryType ) {
	for( unsigned int i = 0; i < RGB; i++ ) {
	    if( mData[i] ) {
		checkCudaErrors( cudaFreeHost( mData[i]) );
		mData[i] = 0;
	    }
	    if( mConvData[i] ) {
		checkCudaErrors( cudaFreeHost( mConvData[i]) );
		mConvData[i] = 0;
	    }
	}
	if( mYComp ) {
	    checkCudaErrors( cudaFreeHost(mYComp) );
	    mYComp = 0;
	}
    } else {
	for( unsigned int i = 0; i < RGB; i++ ) {
	    if( mData[i] ) {
		delete [] mData[i];
		mData[i] = 0;
	    }
	    if( mConvData[i] ) {
		delete [] mConvData[i];
		mConvData[i] = 0;
	    }
	}
	delete [] mYComp;
	mYComp = 0;
    }

    return 0;

}

///
/// Copy a host image by operator=
///
template <typename T>
hsaImage<T> &hsaImage<T>::operator=( const hsaImage<T> &sourceImage )
{

    //
    // Check the size of the source image
    //
    if( sourceImage.mWidth <= 0 || sourceImage.mHeight <= 0 ) {
	std::cerr << "The size of source image must be positive: "
		  << "(" << sourceImage.mWidth << ", " << sourceImage.mHeight << ")"
		  << std::endl;
	exit(1);
    }
    if( mWidth != sourceImage.mWidth || mHeight != sourceImage.mHeight ) {
	std::cerr << "The sizes of the images must be the same: "
		  << "(" << sourceImage.mWidth << ", " << sourceImage.mHeight << "), "
		  << "(" << mWidth << ", " << mHeight << ")" 
		  << std::endl;
	exit(1);
    }

    //
    // Copy the source image
    // Note that memcpy() may be replaced by some functions of C++
    //
    const unsigned int imageSizeByte = sourceImage.mWidth * sourceImage.mHeight * sizeof(T);
    for( unsigned int i = 0; i < RGB; i++ ) {
	memcpy( mData[i], sourceImage.mData[i], imageSizeByte );
	memcpy( mConvData[i], sourceImage.mConvData[i], imageSizeByte );
    }
    memcpy( mYComp, sourceImage.mYComp, imageSizeByte );

    return *this;

}

///
/// Overwrite the original data by the color converted data
///
template <typename T>
void hsaImage<T>::overwriteOriginalDataByConvertedData()
{

    for( unsigned int i = 0; i < mHeight; i++ ) {
	for( unsigned int j = 0; j < mWidth; j++ ) {
	    for( unsigned int k = 0; k < RGB; k++ ) {
		unsigned int pixelPos = i * mWidth + j; 
		mData[k][pixelPos] = mConvData[k][pixelPos];
	    }
	}
    }
    
}

///
/// Get the size of a TIFF image
///
template <typename T>
int hsaImage<T>::tiffGetImageSize( const std::string &sImageFileName )
{

    // 
    // Open TIFF file
    //
    TIFF *tif;
    if( ( tif = TIFFOpen( sImageFileName.c_str(), "r" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }

    //
    // Get the size of the image
    //
    TIFFGetField( tif, TIFFTAG_IMAGEWIDTH, &mWidth );
    TIFFGetField( tif, TIFFTAG_IMAGELENGTH, &mHeight );

    //
    // Close TIFF image
    //
    TIFFClose(tif);

    return 0;

}

///
/// Read a TIFF image
///
template <typename T>
int hsaImage<T>::tiffReadImage( const std::string &sImageFileName )
{

    // 
    // Open the TIFF image
    // 
    TIFF *tif;
    if( ( tif = TIFFOpen( sImageFileName.c_str(), "r" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }

    //
    // Read the image to a 1D array
    //

    // Allocate the memory space
    size_t noPixel;               	// number of pixels
    uint32 *image1d;             	// 1D image
    noPixel = mWidth * mHeight;
    if( ( image1d = (uint32 *)_TIFFmalloc( noPixel * sizeof(uint32) ) ) == NULL ) {
	std::cerr << "Could not allocate the memory space for image1d: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    // Read the image to the 1D array
    if( TIFFReadRGBAImage( tif, mWidth, mHeight, image1d, 0 ) == 0 ) {
	std::cerr << "Could not read the image file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }

    //
    // Copy the image to the object
    //
    unsigned int pixelPos1d,pixelPos2d;	// the pixel positions in the 1D and 2D arrays
    for( unsigned int i = 0; i < mHeight; i++ ) {
	for( unsigned int j = 0; j < mWidth; j++ ) {
	    pixelPos1d = ( ( mHeight - 1 ) - i ) * mWidth + j;
	    pixelPos2d = i * mWidth + j;
	    mData[0][pixelPos2d] = static_cast<T>( image1d[pixelPos1d] & 0x000000FF );		// R
	    mData[1][pixelPos2d] = static_cast<T>(( image1d[pixelPos1d] & 0x0000FF00 ) >> 8); 	// G
	    mData[2][pixelPos2d] = static_cast<T>(( image1d[pixelPos1d] & 0x00FF0000 ) >> 16);	// B
	    // Copy the RGB data to the color converted data as the default
	    for( unsigned int k = 0; k < RGB; k++ )
		mConvData[k][pixelPos2d] = mData[k][pixelPos2d];
	}
    }
    
    //
    // Free memory space
    //
    _TIFFfree(image1d);
    image1d = 0;
    
    //
    // Close the TIFF image
    //
    TIFFClose(tif);

    return 0;

}

///
/// Save a TIFF image
///
template <typename T>
int hsaImage<T>::tiffSaveImage( const std::string &sOutputFileName, const imagePointerNo &pointerNo ) const
{

    //
    // Copy the image to a 1D array of unsinged char
    //

    // Allocate the memory space
    unsigned char *image1d;
    try {
	image1d = new unsigned char[ mWidth * mHeight * RGB ];
    } catch( const std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for a 1D image in saving a TIFF image: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    // Copy the image data
    switch( pointerNo ) {

      case RGB_DATA:
	for( unsigned int i = 0; i < mHeight; i++ )
	    for( unsigned int j = 0; j < mWidth; j++ )
		for( unsigned int k = 0; k < RGB; k++ )
		    image1d[ ( i * mWidth + j ) * RGB + k ] =
			static_cast<unsigned char>(mData[k][ i * mWidth + j ]);
	break;

      case Y_COMPONENT:
	for( unsigned int i = 0; i < mHeight; i++ )
	    for( unsigned int j = 0; j < mWidth; j++ )
		for( unsigned int k = 0; k < RGB; k++ )
		    image1d[ ( i * mWidth + j ) * RGB + k ] =
			static_cast<unsigned char>(mYComp[ i * mWidth + j ]);
	break;

      case CONVERTED_DATA:
	for( unsigned int i = 0; i < mHeight; i++ )
	    for( unsigned int j = 0; j < mWidth; j++ )
		for( unsigned int k = 0; k < RGB; k++ )
		    image1d[ ( i * mWidth + j ) * RGB + k ] =
			static_cast<unsigned char>(mConvData[k][ i * mWidth + j ]);
	break;

      default: // Save the original data
	for( unsigned int i = 0; i < mHeight; i++ )
	    for( unsigned int j = 0; j < mWidth; j++ )
		for( unsigned int k = 0; k < RGB; k++ )
		    image1d[ ( i * mWidth + j ) * RGB + k ] =
			static_cast<unsigned char>(mData[k][ i * mWidth + j ]);
	break;  

    }
    
    //
    // Save the image as a TIFF
    //
    TIFF *tif;
    // Open the TIFF file
    if( ( tif = TIFFOpen( sOutputFileName.c_str(), "w" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sOutputFileName
		  << std::endl;
	exit(1);
    }
    
    // Write the fields
    // Set image size
    TIFFSetField( tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32>(mWidth) );
    TIFFSetField( tif, TIFFTAG_IMAGELENGTH, static_cast<uint32>(mHeight) );
    // Set the data length for a pixel (bytes): RGB
    TIFFSetField( tif, TIFFTAG_SAMPLESPERPIXEL, RGB );
    // Set the data length for each info (bits): 8bit = 256 level
    TIFFSetField( tif, TIFFTAG_BITSPERSAMPLE, 8 );
    // Set the origin of the image
    TIFFSetField( tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT );
    // Set the compression option
    TIFFSetField( tif, TIFFTAG_COMPRESSION, 1 );
    // Set the other fields
    TIFFSetField( tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG );
    TIFFSetField( tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB );
    TIFFSetField( tif, TIFFTAG_ROWSPERSTRIP, mHeight );

    // Save the image
    if( TIFFWriteEncodedStrip( tif, 0, image1d, mWidth * mHeight * RGB ) < 0 ) {
	std::cerr << "Could not write the image file: "
		  << sOutputFileName
		  << std::endl;
	exit(1);
    }

    // Close the TIFF file
    TIFFClose(tif);

    //
    // Delete the memory space
    //
    delete [] image1d;
    image1d = 0;

    return 0;

}

///
/// Get the size of a JPEG image
///
template <typename T>
int hsaImage<T>::jpegGetImageSize( const std::string &sImageFileName )
{

    //
    // Initialize JPEG structures
    //
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error( &jerr );
    jpeg_create_decompress( &cinfo );

    //
    // Open the image file
    //
    FILE *fp;
    if( ( fp = fopen( sImageFileName.c_str(), "r" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }

    //
    // Set the source
    //
    jpeg_stdio_src( &cinfo, fp );

    //
    // Set the header
    //
    jpeg_read_header( &cinfo, TRUE );

    //
    // Start decompression
    // 
    jpeg_start_decompress( &cinfo );

    //
    // Get the width and height of the image
    //
    mWidth = cinfo.output_width;
    mHeight = cinfo.output_height;

    //
    // The following code reads the image
    // It's required to read the size of the image normally.
    // (might not be required)
    //
    
    //
    // Allocate the memory space for the image
    //
    JSAMPARRAY img;
    try {
	img = new JSAMPROW[mHeight];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for img: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i = 0; i < mHeight; i++ ) {
	try {
	    img[i] = new JSAMPLE[ mWidth * RGB ];
	} catch( std::bad_alloc & ) {
	    std::cerr << "Could not allocate the memory space for img[" << i << "]: "
		      << __FILE__ << " : " << __LINE__
		      << std::endl;
	    exit(1);
	}
    }
    
    //
    // Get image data
    //
    while( cinfo.output_scanline < cinfo.output_height )
	jpeg_read_scanlines( &cinfo, img + cinfo.output_scanline, cinfo.output_height - cinfo.output_scanline );

    //
    // Stop decmpression
    //
    jpeg_finish_decompress( &cinfo );

    //
    // Destroy the JPEG structure
    //
    jpeg_destroy_decompress( &cinfo );

    //
    // Close the image file
    //
    fclose(fp);

    //
    // Delete the memory space
    //
    for( unsigned int i = 0; i < mHeight; i++ ) {
	delete [] img[i];
	img[i] = 0;
    }
    delete img;
    img = 0;
    
    return 0;

}

///
/// Read a JPEG image
///
template <typename T>
int hsaImage<T>::jpegReadImage( const std::string &sImageFileName )
{

    //
    // Initialize JPEG structures
    //
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error( &jerr );
    jpeg_create_decompress( &cinfo );

    //
    // Open the image file
    //
    FILE *fp;
    if( ( fp = fopen( sImageFileName.c_str(), "r" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }

    //
    // Set the source
    //
    jpeg_stdio_src( &cinfo, fp );

    //
    // Set the header
    //
    jpeg_read_header( &cinfo, TRUE );

    //
    // Start decompression
    // 
    jpeg_start_decompress( &cinfo );

    //
    // Read the image
    //
    JSAMPARRAY img;
    if( cinfo.output_components == RGB ) { // color image 

	// Allocate the memory space for the image
	try {
	    img = new JSAMPROW[mHeight];
	} catch( std::bad_alloc & ) {
	    std::cerr << "Could not allocate the memory space for img: "
		      << __FILE__ << " : " << __LINE__
		      << std::endl;
	    exit(1);
	}
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    try {
		img[i] = new JSAMPLE[ mWidth * RGB ];
	    } catch( std::bad_alloc & ) {
		std::cerr << "Could not allocate the memory space for img[" << i << "]: "
			  << __FILE__ << " : " << __LINE__
			  << std::endl;
		exit(1);
	    }
	}

	// Get the image data
	while( cinfo.output_scanline < cinfo.output_height )
	    jpeg_read_scanlines( &cinfo, img + cinfo.output_scanline, cinfo.output_height - cinfo.output_scanline );

	// Stop decmpression
	jpeg_finish_decompress( &cinfo );

	// Destroy the JPEG structure
	jpeg_destroy_decompress( &cinfo );

	// Close the image file
	fclose(fp);

	// Copy the image to the object
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		mData[0][pixelPos] = static_cast<T>(img[i][ j * RGB ]);
		mData[1][pixelPos] = static_cast<T>(img[i][ j * RGB + 1 ]);
		mData[2][pixelPos] = static_cast<T>(img[i][ j * RGB + 2 ]);
		// Copy the RGB data to the color converted data as the default
		for( unsigned int k = 0; k < RGB; k++ )
		    mConvData[k][pixelPos] = mData[k][pixelPos];
	    }
	}

    } else if( cinfo.output_components == 1 ) { // greyscale image

	// Allocate the memory space for the image
	try {
	    img = new JSAMPROW[mHeight];
	} catch( std::bad_alloc & ) {
	    std::cerr << "Could not allocate the memory space for img: "
		      << __FILE__ << " : " << __LINE__
		      << std::endl;
	    exit(1);
	}
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    try {
		img[i] = new JSAMPLE[mWidth];
	    } catch( std::bad_alloc & ) {
		std::cerr << "Could not allocate the memory space for img[" << i << "]: "
			  << __FILE__ << " : " << __LINE__
			  << std::endl;
		exit(1);
	    }
	}

	// Get the image data
	while( cinfo.output_scanline < cinfo.output_height )
	    jpeg_read_scanlines( &cinfo, img + cinfo.output_scanline, cinfo.output_height - cinfo.output_scanline );

	// Stop decmpression
	jpeg_finish_decompress( &cinfo );

	// Destroy the JPEG structure
	jpeg_destroy_decompress( &cinfo );

	// Close the image file
	fclose(fp);

	// Copy the image to the object
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		mData[0][pixelPos] = static_cast<T>(img[i][j]);
		mData[1][pixelPos] = static_cast<T>(img[i][j]);
		mData[2][pixelPos] = static_cast<T>(img[i][j]);
		// Copy the RGB data to the color converted data as the default
		for( unsigned int k = 0; k < RGB; k++ )
		    mConvData[k][pixelPos] = mData[k][pixelPos];
	    }
	}

    }
	
    //
    // Delete the memory space
    //
    for( unsigned int i = 0; i < mHeight; i++ ) {
	delete [] img[i];
	img[i] = 0;
    }
    delete img;
    img = 0;

    return 0;

}

///
/// Save a JPEG image
///
template <typename T>
int hsaImage<T>::jpegSaveImage( const std::string &sOutputFileName, const imagePointerNo &pointerNo ) const
{

    //
    // Allocate memory space for the image 
    //
    JSAMPARRAY img;
    try {
	img = new JSAMPROW[mHeight];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for img: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i = 0; i < mHeight; i++ ) {
	try {
	    img[i] = new JSAMPLE[ mWidth * RGB ];
	} catch( std::bad_alloc & ) {
	    std::cerr << "Could not allocate the memory space for img[" << i << "]: "
		      << __FILE__ << " : " << __LINE__
		      << std::endl;
	    exit(1);
	}
    }
    
    //
    // Initailize JPEG structures
    //
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error( &jerr );
    jpeg_create_compress( &cinfo );

    //
    // Open the image file
    //
    FILE *fp;
    if( ( fp = fopen( sOutputFileName.c_str(), "w" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sOutputFileName
		  << std::endl;
	exit(1);
    }

    //
    // Set the destination
    //
    jpeg_stdio_dest( &cinfo, fp );

    //
    // Set the parameters
    //
    cinfo.image_width = mWidth;
    cinfo.image_height = mHeight;
    cinfo.input_components = RGB;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults( &cinfo );

    //
    // Start compression
    //
    jpeg_start_compress( &cinfo, TRUE );

    //
    // Save the image
    //
    switch( pointerNo ) {

      case RGB_DATA:
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		img[i][ j * RGB ] = static_cast<unsigned char>(mData[0][pixelPos]);
		img[i][ j * RGB + 1 ] = static_cast<unsigned char>(mData[1][pixelPos]);
		img[i][ j * RGB + 2 ] = static_cast<unsigned char>(mData[2][pixelPos]);
	    }
	}
	break;

      case Y_COMPONENT:
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		img[i][ j * RGB ] = static_cast<unsigned char>(mYComp[pixelPos]);
		img[i][ j * RGB + 1 ] = static_cast<unsigned char>(mYComp[pixelPos]);
		img[i][ j * RGB + 2 ] = static_cast<unsigned char>(mYComp[pixelPos]);
	    }
	}
	break;

      case CONVERTED_DATA:
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		img[i][ j * RGB ] = static_cast<unsigned char>(mConvData[0][pixelPos]);
		img[i][ j * RGB + 1 ] = static_cast<unsigned char>(mConvData[1][pixelPos]);
		img[i][ j * RGB + 2 ] = static_cast<unsigned char>(mConvData[2][pixelPos]);
	    }
	}
	break;

      default: // Save the original data
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		img[i][ j * RGB ] = static_cast<unsigned char>(mData[0][pixelPos]);
		img[i][ j * RGB + 1 ] = static_cast<unsigned char>(mData[1][pixelPos]);
		img[i][ j * RGB + 2 ] = static_cast<unsigned char>(mData[2][pixelPos]);
	    }
	}
	break;

    }

    jpeg_write_scanlines( &cinfo, img, mHeight );

    //
    // Stop compression
    //
    jpeg_finish_compress( &cinfo );

    //
    // Destory the JPEG structure
    //
    jpeg_destroy_compress( &cinfo );

    //
    // Close the image file
    //
    fclose(fp);

    //
    // Delete the memory space
    //
    for( unsigned int i = 0; i < mHeight; i++ ) {
	delete [] img[i];
	img[i] = 0;
    }
    delete img;
    img = 0;

    return 0;

}

///
/// Get the size of a PNG image
///
template <typename T>
int hsaImage<T>::pngGetImageSize( const std::string &sImageFileName )
{

    //
    // Create structures for a PNG image
    //
    png_structp pngPtr;
    png_infop infoPtr;

    if( ( pngPtr = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL ) ) == NULL ) {
	std::cerr << "Could not allocate pngPtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    if( ( infoPtr = png_create_info_struct( pngPtr ) ) == NULL ) {
	std::cerr << "Could not allocate infoPtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	png_destroy_read_struct( &pngPtr, (png_infopp)NULL, (png_infopp)NULL );
	exit(1);
    }

    //
    // Open the image file
    //
    FILE *fp;
    if( ( fp = fopen( sImageFileName.c_str(), "r" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }
    
    //
    // Check if the file is a PNG image
    //
    png_byte sig[PNG_BYTES_TO_CHECK]; // signature data for check
    if( fread( sig, 1, PNG_BYTES_TO_CHECK, fp ) != PNG_BYTES_TO_CHECK ) { // read signature data
	std::cerr << "Could not read the signature of the PNG file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }
    if( png_sig_cmp( sig, 0, PNG_BYTES_TO_CHECK ) ) { // check the signature data
	std::cerr << "The file is not a PNG image: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }

    //
    // Get the buffers for error handling
    //
    if( setjmp( png_jmpbuf(pngPtr) ) ) {
	png_destroy_read_struct( &pngPtr, &infoPtr, (png_infopp)NULL );
	exit(1);
    }
    
    //
    // Read info.
    //
    png_init_io( pngPtr, fp );
    png_set_sig_bytes( pngPtr, PNG_BYTES_TO_CHECK );
    png_read_info( pngPtr, infoPtr );

    //
    // Set the size of the image
    //
    mWidth = png_get_image_width( pngPtr, infoPtr );
    mHeight = png_get_image_height( pngPtr, infoPtr );

    //
    // Close the image
    //
    fclose(fp);

    //
    // Free the strucutre for a PNG image
    //
    png_destroy_read_struct( &pngPtr, &infoPtr, (png_infopp)NULL );

    return 0;

}

template <typename T>
int hsaImage<T>::pngReadImage( const std::string &sImageFileName )
{

    //
    // Create structures for a PNG image
    //
    png_structp pngPtr;
    png_infop infoPtr;

    if( ( pngPtr = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL ) ) == NULL ) {
	std::cerr << "Could not allocate pngPtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    if( ( infoPtr = png_create_info_struct( pngPtr ) ) == NULL ) {
	std::cerr << "Could not allocate infoPtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	png_destroy_read_struct( &pngPtr, (png_infopp)NULL, (png_infopp)NULL );
	exit(1);
    }

    //
    // Open the image
    //
    FILE *fp;
    if( ( fp = fopen( sImageFileName.c_str(), "r" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }

    //
    // Check if the image is a PNG file
    //
    png_byte sig[PNG_BYTES_TO_CHECK]; // signature data for check
    if( fread( sig, 1, PNG_BYTES_TO_CHECK, fp ) != PNG_BYTES_TO_CHECK ) { // read signature data
	std::cerr << "Could not read the signature of the PNG file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }
    if( png_sig_cmp( sig, 0, PNG_BYTES_TO_CHECK ) ) { // check signature data
	std::cerr << "The image file is not a PNG file: "
		  << sImageFileName
		  << std::endl;
	exit(1);
    }

    //
    // Set the buffers for error handling
    //
    if( setjmp( png_jmpbuf(pngPtr) ) ) {
	std::cerr << "Error during init_io."
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	png_destroy_read_struct( &pngPtr, &infoPtr, (png_infopp)NULL );
	exit(1);
    }

    //
    // Read info.
    //
    png_init_io( pngPtr, fp );
    png_set_sig_bytes( pngPtr, PNG_BYTES_TO_CHECK );
    png_read_info( pngPtr, infoPtr );

    //
    // Allocate the memory space for a temporary image
    //
    png_bytepp img;
    try {
	img = new png_bytep[mHeight];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for img: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i = 0; i < mHeight; i++ ) {
	try {
	    img[i] = new png_byte[ mWidth * RGB ];
	} catch( std::bad_alloc & ) {
	    std::cerr << "Could not allocate the memory space for img[" << i << "]: "
		      << __FILE__ << " : " << __LINE__
		      << std::endl;
	    exit(1);
	}
    }
    
    //
    // Read the image
    //
    png_read_image( pngPtr, img );

    //
    // Copy the image
    //
    for( unsigned int i = 0; i < mHeight; i++ ) {
	for( unsigned int j = 0; j < mWidth; j++ ) {
	    unsigned int pixelPos = i * mWidth + j;
	    mData[0][pixelPos] = static_cast<T>(img[i][ j * RGB ]);
	    mData[1][pixelPos] = static_cast<T>(img[i][ j * RGB + 1 ]);
	    mData[2][pixelPos] = static_cast<T>(img[i][ j * RGB + 2 ]);
	    // Copy the RGB data to the color converted data as the default
	    for( unsigned int k = 0; k < RGB; k++ )
		mConvData[k][pixelPos] = mData[k][pixelPos];
	}
    }

    //
    // Close the image
    //
    fclose(fp);

    //
    // Destroy the structure
    //
    png_destroy_read_struct( &pngPtr, &infoPtr, (png_infopp)NULL );

    //
    // Delete the memory space
    //
    for( unsigned int i = 0; i < mHeight; i++ ) {
	delete [] img[i];
	img[i] = 0;
    }
    delete img;
    img = 0;
    
    return 0;

}

template <typename T>
int hsaImage<T>::pngSaveImage( const std::string &sOutputFileName, const imagePointerNo &pointerNo ) const
{

    //
    // Create structures for a PNG image
    //
    png_structp pngPtr;
    png_infop infoPtr;
    
    if( ( pngPtr = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL ) ) == NULL ) {
	std::cerr << "Could not allocate pngPtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    if( ( infoPtr = png_create_info_struct( pngPtr ) ) == NULL ) {
	std::cerr << "Could not allocate infoPtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	png_destroy_read_struct( &pngPtr, (png_infopp)NULL, (png_infopp)NULL );
	exit(1);
    }

    //
    // Allocate the memory space a temporary image
    //
    png_bytepp img;
    try {
	img = new png_bytep[mHeight];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for img: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i = 0; i < mHeight; i++ ) {
	try {
	    img[i] = new png_byte[ mWidth * RGB ];
	} catch( std::bad_alloc & ) {
	    std::cerr << "Could not allocate the memory space for img[" << i << "]: "
		      << __FILE__ << " : " << __LINE__
		      << std::endl;
	    exit(1);
	}
    }
    
    //
    // Open the image
    //
    FILE *fp;
    if( ( fp = fopen( sOutputFileName.c_str(), "w" ) ) == NULL ) {
	std::cerr << "Could not open the image file: "
		  << sOutputFileName
		  << std::endl;
	exit(1);
    }

    //
    // Set IO
    //

    // Set the buffers for error handling
    if( setjmp( png_jmpbuf(pngPtr) ) ) {
	std::cerr << "Error during init_io:"
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	png_destroy_read_struct( &pngPtr, &infoPtr, (png_infopp)NULL );
	exit(1);
    }
    // Set IO
    png_init_io( pngPtr, fp );

    //
    // Write header
    //

    // Set the buffers for error handling
    if( setjmp( png_jmpbuf(pngPtr) ) ) {
	std::cerr << "Error during writing the header: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	png_destroy_read_struct( &pngPtr, &infoPtr, (png_infopp)NULL );
	exit(1);
    }
    // Write info.
    png_set_IHDR( pngPtr, infoPtr, mWidth, mHeight, DEFAULT_BIT_DEPTH, DEFAULT_COLOR_TYPE, PNG_INTERLACE_NONE,
		  PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT );
    png_write_info( pngPtr, infoPtr );

    //
    // Write the image data
    //

    // Set buffers for error handling
    if( setjmp( png_jmpbuf(pngPtr) ) ) {
	std::cerr << "Error during writing the image: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	png_destroy_read_struct( &pngPtr, &infoPtr, (png_infopp)NULL );
	exit(1);
    }

    // Copy image data
    switch( pointerNo ) {

      case RGB_DATA:
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		img[i][ j * RGB ] = static_cast<unsigned char>(mData[0][pixelPos]);
		img[i][ j * RGB + 1 ] = static_cast<unsigned char>(mData[1][pixelPos]);
		img[i][ j * RGB + 2 ] = static_cast<unsigned char>(mData[2][pixelPos]);
	    }
	}
	break;

      case Y_COMPONENT:
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		img[i][ j * RGB ] = static_cast<unsigned char>(mYComp[pixelPos]);
		img[i][ j * RGB + 1 ] = static_cast<unsigned char>(mYComp[pixelPos]);
		img[i][ j * RGB + 2 ] = static_cast<unsigned char>(mYComp[pixelPos]);
	    }
	}
	break;

      case CONVERTED_DATA:
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		img[i][ j * RGB ] = static_cast<unsigned char>(mConvData[0][pixelPos]);
		img[i][ j * RGB + 1 ] = static_cast<unsigned char>(mConvData[1][pixelPos]);
		img[i][ j * RGB + 2 ] = static_cast<unsigned char>(mConvData[2][pixelPos]);
	    }
	}
	break;

      default: // Save the original data
	for( unsigned int i = 0; i < mHeight; i++ ) {
	    for( unsigned int j = 0; j < mWidth; j++ ) {
		unsigned int pixelPos = i * mWidth + j;
		img[i][ j * RGB ] = static_cast<unsigned char>(mData[0][pixelPos]);
		img[i][ j * RGB + 1 ] = static_cast<unsigned char>(mData[1][pixelPos]);
		img[i][ j * RGB + 2 ] = static_cast<unsigned char>(mData[2][pixelPos]);
	    }
	}
	break;
    }

    // Write the image
    png_write_image( pngPtr, img );

    //
    // Check the end of writing
    //

    // Set buffers for error handling
    if( setjmp( png_jmpbuf(pngPtr) ) ) {
	std::cerr << "Error during writing end info: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	png_destroy_read_struct( &pngPtr, &infoPtr, (png_infopp)NULL );
	exit(1);
    }
    // Write end info.
    png_write_end( pngPtr, NULL );

    //
    // Close the image
    //
    fclose(fp);

    //
    // Destroy the structure
    //
    png_destroy_write_struct( &pngPtr, &infoPtr );

    //
    // Delete the memory spaces
    //
    for( unsigned int i = 0; i < mHeight; i++ ) {
	delete [] img[i];
	img[i] = 0;
    }
    delete img;
    img = 0;
    
    return 0;

}

///
/// Saving pixel value
/// The file format is for gnuplot.
///
template <typename T>
int hsaImage<T>::outputYCompImagePixelValues( const std::string &sOutputFileName ) const
{

    //
    // Open the output file
    //
    std::ofstream fout;
    fout.open( sOutputFileName.c_str() );
    if( !fout ) {
	std::cerr << "Could not open the image file: "
		  << sOutputFileName
		  << std::endl;
	exit(1);
    }

    //
    // Output pixle values
    //
    for( unsigned int i = 0; i < mHeight; i++ ) {	
	for( unsigned int j = 0; j < mWidth; j++ ) {
	    fout << mYComp[ ( ( mHeight - 1 ) - i ) * mWidth + j ] << std::endl; // flip the vertical axis
	}
	fout << std::endl;
    }

    //
    // Close the output file
    //
    fout.close();

    return 0;

}

////
//// Explicit instantiation for the template class.
//// The member functions are also instantiated by explicit instantiation.
//// The type of the template can be predefined.
////
template class dsaImage<float>;
template class dsaImage<double>;

template class hsaImage<unsigned char>;
template class hsaImage<float>;
template class hsaImage<double>;

