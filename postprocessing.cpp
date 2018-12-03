////
//// postprocessing.cpp: the functions for postprocessing of images
////

///
/// The standard include files
///
#include <iostream>
#include <limits>

#include <cmath>

///
/// The include file for postprocessing
///
#include "image_rw_cuda.h"

///
/// The function for taking the absolute value of each pixel value
///
template <typename T>
int takeImageAbsoluteValueCPU( hsaImage<T> *h_image, const unsigned int &noChannels )
{

    //
    // Set the pointers of the image
    //
    T **imagePtr;
    try {
	imagePtr = new T *[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the momery space for imagePtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    if( noChannels == 3 ) // RGB image
	for( unsigned int i = 0; i < noChannels; i++ )
	    imagePtr[i] = h_image->getImagePtr( i );
    else if( noChannels == 1 ) // Y Component
	imagePtr[0] = h_image->getYCompPtr();

    //
    // Take the absolute value of each pixel value
    //
    for( unsigned int i = 0; i < noChannels; i++ ) {
	for( unsigned int j = 0; j < h_image->getImageHeight(); j++ ) {
	    for( unsigned int k = 0; k < h_image->getImageWidth(); k++ ) {
		unsigned int pixelPos = j * h_image->getImageWidth() + k;
		imagePtr[i][pixelPos] = fabs( imagePtr[i][pixelPos] );
	    }
	}
    }

    return 0;

}

///
/// The function for normalizing image
///
template <typename T>
int normalizeImageCPU( hsaImage<T> *h_image, const unsigned int &noChannels )
{

    //
    // Set the pointers of the image
    //
    T **imagePtr;
    try {
	imagePtr = new T *[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the momery space for imagePtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    if( noChannels == 3 ) // RGB image
	for( unsigned int i = 0; i < noChannels; i++ )
	    imagePtr[i] = h_image->getImagePtr( i );
    else if( noChannels == 1 ) // Y Component
	imagePtr[0] = h_image->getYCompPtr();

    //
    // Find the maximum and minimum of the pixel values
    //
    T *maxImageValue;
    try {
	maxImageValue = new T[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for maxImageValue: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i =0; i < noChannels; i++ )
	maxImageValue[i] = std::numeric_limits<T>::min();
    
    T *minImageValue;
    try {
	minImageValue = new T[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for minImageValue: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i =0; i < noChannels; i++ )
	minImageValue[i] = std::numeric_limits<T>::max();

    for( unsigned int i = 0; i < noChannels; i++ ) {
	for( unsigned int j = 0; j < h_image->getImageHeight(); j++ ) {
	    for( unsigned int k = 0; k < h_image->getImageWidth(); k++ ) {
		unsigned int pixelPos = j * h_image->getImageWidth() + k;
		if( imagePtr[i][pixelPos] > maxImageValue[i] )
		    maxImageValue[i] = imagePtr[i][pixelPos];
		if( imagePtr[i][pixelPos] < minImageValue[i] )
		    minImageValue[i] = imagePtr[i][pixelPos];
	    }
	}
    }

    //
    // Normalize the image
    //
    const T maxPixelValue = 255.0;
    for( unsigned int i = 0; i < noChannels; i++ ) {
	T imageValueInterval = maxImageValue[i] - minImageValue[i]; 
	for( unsigned int j = 0; j < h_image->getImageHeight(); j++ ) {
	    for( unsigned int k = 0; k < h_image->getImageWidth(); k++ ) {
		unsigned int pixelPos = j * h_image->getImageWidth() + k;
		imagePtr[i][pixelPos] = maxPixelValue * ( imagePtr[i][pixelPos] - minImageValue[i] ) / imageValueInterval;
	    }
	}
    }

    //
    // Delete the memory space
    //
    delete [] minImageValue;
    minImageValue = 0;
    delete [] maxImageValue;
    maxImageValue = 0;

    return 0;

}

///
/// The function for level adjustment	
///
template <typename T>
int adjustImageLevelCPU( hsaImage<T> *h_image, const unsigned int &noChannels, const T &maxLevel )
{

    //
    // Set the pointers of the image
    //
    T **imagePtr;
    try {
	imagePtr = new T *[noChannels];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the momery space for imagePtr: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    if( noChannels == 3 ) // RGB image
	for( unsigned int i = 0; i < noChannels; i++ )
	    imagePtr[i] = h_image->getImagePtr( i );
    else if( noChannels == 1 ) // Y Component
	imagePtr[0] = h_image->getYCompPtr();

    //
    // Adjust the level of the image
    //
    const T maxPixelValue = 255.0;
    for( unsigned int i = 0; i < noChannels; i++ ) {
	for( unsigned int j = 0; j < h_image->getImageHeight(); j++ ) {
	    for( unsigned int k = 0; k < h_image->getImageWidth(); k++ ) {
		unsigned int pixelPos = j * h_image->getImageWidth() + k;
		if( imagePtr[i][pixelPos] > maxLevel )
		    imagePtr[i][pixelPos] = maxPixelValue;
		else
		    imagePtr[i][pixelPos] =  imagePtr[i][pixelPos] * maxPixelValue / maxLevel;
	    }
	}
    }

    return 0;

}

////
//// Explicit instantiation of the template functions
////
template
int takeImageAbsoluteValueCPU( hsaImage<float> *h_image, const unsigned int &noChannels );
template
int takeImageAbsoluteValueCPU( hsaImage<double> *h_image, const unsigned int &noChannels );

template
int normalizeImageCPU( hsaImage<float> *h_image, const unsigned int &noChannels );
template
int normalizeImageCPU( hsaImage<double> *h_image, const unsigned int &noChannels );

template
int adjustImageLevelCPU( hsaImage<float> *h_image, const unsigned int &noChannels, const float &maxLevel );
template
int adjustImageLevelCPU( hsaImage<double> *h_image, const unsigned int &noChannels, const double &maxLevel );
