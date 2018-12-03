////
//// padding.cpp: perform image padding for filtering
////

///
/// The standard include files
///
#include <iostream>

#include <cstdlib>

///
/// The include files for padding
///
#include "image_rw_cuda.h"
#include "padding.h"

///
/// Replication padding: the values on borders are replicated to the outside of an image
///
template <typename T>
int replicationPadding( T *image, const unsigned int &iWidth, const unsigned int &iHeight,
			const int &hFilterSize, 
			T *paddedImage, const unsigned int &paddedIWidth, const unsigned int &paddedIHeight )
{

    //
    // Perform extended padding
    //
    for( unsigned int i = 0; i < paddedIHeight; i++ ) {
	for( unsigned int j = 0; j < paddedIWidth; j++ ) {

	    // Set the pixel position of the extended image
	    unsigned int extendedPixelPos = i * paddedIWidth + j;

	    // Set the pixel position of the input image
	    unsigned int pixelPos;
	    if( j >= 0 && j < hFilterSize && 
		i >= 0 && i < hFilterSize ) { // (0,0)
		pixelPos = 0; 
	    } else if( j >= hFilterSize && j < iWidth + hFilterSize &&
		       i >= 0 && i < hFilterSize ) { // (u,0)
		pixelPos = j - hFilterSize;
	    } else if( j >= iWidth + hFilterSize && j < iWidth + 2 * hFilterSize &&
		       i >= 0 && i < hFilterSize ) { // (iWidth-1,0)
		pixelPos = iWidth - 1;
	    } else if( j >= 0 && j < hFilterSize &&
		       i >= hFilterSize && i < iHeight + hFilterSize ) { // (0,v)
		pixelPos = ( i - hFilterSize ) * iWidth;
	    } else if( j >= hFilterSize && j < iWidth + hFilterSize &&
		       i >= hFilterSize && i < iHeight + hFilterSize ) { // (u,v)
		pixelPos = ( i - hFilterSize ) * iWidth + ( j - hFilterSize );
	    } else if( j >= iWidth + hFilterSize && j < iWidth + 2 * hFilterSize &&
		       i >= hFilterSize && i < iHeight + hFilterSize ) { // (iWidth-1,v)
		pixelPos = ( i - hFilterSize ) * iWidth + ( iWidth - 1 );
	    } else if( j >= 0 && j < hFilterSize &&
		       i >= iHeight + hFilterSize && i < iHeight + 2 * hFilterSize ) { // (0,iHeight-1)
		pixelPos = ( iHeight - 1 ) * iWidth;
	    } else if( j >= hFilterSize && j < iWidth + hFilterSize &&
		       i >= iHeight + hFilterSize && i < iHeight + 2 * hFilterSize ) { // (u,iHeight-1)
		pixelPos = ( iHeight - 1 ) * iWidth + ( j - hFilterSize );
	    } else if( j >= iWidth + hFilterSize && j < iWidth + 2 * hFilterSize &&
		       i >= iHeight + hFilterSize && i < iHeight + 2 * hFilterSize ) { // (iWidth-1,iHeight-1)
		pixelPos = ( iHeight - 1 ) * iWidth + ( iWidth - 1 );
	    }

	    // Copy the pixel value
	    paddedImage[extendedPixelPos] = image[pixelPos]; 

	}
    }
    
    return 0;

}

///
/// zero padding: the outside of an image is padded with zero
///
template <typename T>
int zeroPadding( T *fmap, const unsigned int &fmapWidth, const unsigned int &fmapHeight,
		 const int &hFilterSize, 
		 T *paddedFmap, const unsigned int &paddedFmapWidth, const unsigned int &paddedFmapHeight )
{

    //
    // Perform padded padding
    //
    for( unsigned int i = 0; i < paddedFmapHeight; i++ ) {
	for( unsigned int j = 0; j < paddedFmapWidth; j++ ) {

	    // Set the pixel position of the padded fmap
	    unsigned int paddedPixelPos = i * paddedFmapWidth + j;

	    // Copy the pixel value
	    if( i >= hFilterSize && i < fmapHeight + hFilterSize &&
		j >= hFilterSize && j < fmapWidth + hFilterSize ) {
		unsigned int pixelPos = ( i - hFilterSize ) * fmapWidth + ( j - hFilterSize );
		paddedFmap[paddedPixelPos] = fmap[pixelPos];
	    } else {
		paddedFmap[paddedPixelPos] = 0.0;
	    }

	}
    }
    
    return 0;

}

///
/// Pad an image for filtering 
///
template <typename T>
int imagePadding( hsaImage<T> &image, const unsigned int &filterSize, const unsigned int &paddingType,
		  hsaImage<T> *paddedImage )
{

    //
    // Set the pointers for the image
    //
    const unsigned int RGB = 3;
    T *imagePtr[RGB];
    for( unsigned int i = 0; i < RGB; i++ )
	imagePtr[i] = image.getImagePtr( i );
    T *paddedImagePtr[RGB];
    for( unsigned int i = 0; i < RGB; i++ )
	paddedImagePtr[i] = paddedImage->getImagePtr( i );

    //
    // Set the half of the fitler size
    //
    unsigned int hFilterSize = filterSize / 2;

    //
    // Perform replication padding
    //
    for( unsigned int i = 0; i < RGB; i++ ) {
	if( paddingType == ZERO_PADDING ) {
	    zeroPadding( imagePtr[i], image.getImageWidth(), image.getImageHeight(),
			 hFilterSize,
			 paddedImagePtr[i], paddedImage->getImageWidth(), paddedImage->getImageHeight() );
	} else if( paddingType == REPLICATION_PADDING ) {
	    replicationPadding( imagePtr[i], image.getImageWidth(), image.getImageHeight(),
				hFilterSize,
				paddedImagePtr[i], paddedImage->getImageWidth(), paddedImage->getImageHeight() );
	} else {
	    std::cerr << "The padding type must be either ZERO_PADDING or REPLICATION_PADDING: "
		      << __FILE__ << " : " << __LINE__
		      << std::endl;
	    exit(1);
	}
    }
    
    return 0;

}

////
//// Explicit instantiation
////
template
int replicationPadding( float *image, const unsigned int &iWidth, const unsigned int &iHeight,
			const int &hFilterSize, 
			float *paddedImage, const unsigned int &paddedIWidth, const unsigned int &paddedIHeight );
template
int replicationPadding( double *image, const unsigned int &iWidth, const unsigned int &iHeight,
			const int &hFilterSize, 
			double *paddedImage, const unsigned int &paddedIWidth, const unsigned int &paddedIHeight );

template
int zeroPadding( float *fmap, const unsigned int &fmapWidth, const unsigned int &fmapHeight,
		 const int &hFilterSize, 
		 float *paddedFmap, const unsigned int &paddedFmapWidth, const unsigned int &paddedFmapHeight );
template
int zeroPadding( double *fmap, const unsigned int &fmapWidth, const unsigned int &fmapHeight,
		 const int &hFilterSize, 
		 double *paddedFmap, const unsigned int &paddedFmapWidth, const unsigned int &paddedFmapHeight );

template
int imagePadding( hsaImage<float> &image, const unsigned int &filterSize, const unsigned int &paddingType, hsaImage<float> *paddedImage );
template
int imagePadding( hsaImage<double> &image, const unsigned int &filterSize, const unsigned int &paddingType, hsaImage<double> *paddedImage );
