#ifndef PADDING_H
#define PADDING_H

///
/// Padding type
///
enum paddingType {
    ZERO_PADDING,
    REPLICATION_PADDING
};

///
/// Function prototypes
///
template <typename T>
int replicationPadding( T *image, const unsigned int &iWidth, const unsigned int &iHeight,
			const int &hFilterSize, 
			T *paddedImage, const unsigned int &paddedIWidth, const unsigned int &paddedIHeight );

template <typename T>
int zeroPadding( T *fmap, const unsigned int &iWidth, const unsigned int &iHeight,
		 const int &hFilterSize, 
		 T *paddedImage, const unsigned int &paddedIWidth, const unsigned int &paddedIHeight );

template <typename T>
int imagePadding( hsaImage<T> &image, const unsigned int &filterSize, const unsigned int &paddingType,
		  hsaImage<T> *paddedImage );

#endif // PADDING_H
