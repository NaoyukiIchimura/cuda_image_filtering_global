#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

//
// Function prototypes
//
template <typename T>
int takeImageAbsoluteValueCPU( hsaImage<T> *h_image, const unsigned int &noChannels );

template <typename T>
int normalizeImageCPU( hsaImage<T> *h_image, const unsigned int &noChannels );

template <typename T>
int adjustImageLevelCPU( hsaImage<T> *h_image, const unsigned int &noChannels, const T &maxLevel );

//---

template <typename T>
int takeImageAbsoluteValueGPU( dsaImage<T> *d_image, const unsigned int &noChannels );

template <typename T>
int normalizeImageGPU( dsaImage<T> *d_image, const unsigned int &noChannels );

template <typename T>
int adjustImageLevelGPU( dsaImage<T> *d_image, const unsigned int &noChannels, const T &maxLevel );

#endif // POSTPROCESSING_H
