#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageSmooth.h"
#include "ImageSmoothCV.h"

///----------------------------------------------------------
/// Copy from our buffer to Ipl image buffer taking care of the alignment
///
static void CopyToIplBuffer(unsigned char *buffer, cv::Mat img)
{
  memcpy(img.data, buffer, img.cols*img.rows);
} //end-IplImage


///----------------------------------------------------------
/// Copy from Ipl image buffer taking our buffer care of the alignment
///
static void CopyFromIplBuffer(unsigned char *buffer, cv::Mat img)
{
  memcpy(buffer, img.data, img.cols*img.rows);
} //end-CopyFromIplBuffer

///---------------------------------------------------------------------------------------------------------------------------------------
/// Given an image of size widthxheight in srcImg, smooths the image using a gaussian filter (cvSmooth) and copies the smoothed image to smoothImg
/// If sigma=0.0, then calls cvSmooth(srcImg, smoothedImg, CV_GAUSSIAN, 5, 5); This is the default.
/// If sigma>0.0, then calls cvSmooth(srcImg, smoothedImg, CV_GAUSSIAN, 0, 0, sigma);
///
void SmoothImage(unsigned char *srcImg, unsigned char *smoothImg, int width, int height, double sigma)
{
  if (sigma <= 0){
    memcpy(smoothImg, srcImg, width*height);
    return;
  } //end-if

  cv::Mat img1{width, height, CV_8UC1, srcImg};
  cv::Mat img2{width, height, CV_8UC1, smoothImg};

  if (sigma == 1.0)
    cv::GaussianBlur(img1, img2, cv::Size(5,5), 1.25);
  else if (sigma == 1.5)
    cv::GaussianBlur(img1, img2, cv::Size(7,7), 1.25);
  else
    cv::GaussianBlur(img1, img2, cv::Size(0,0), sigma);

} //end-SmoothImage

///---------------------------------------------------------------------------------------------------------------------------------------
/// Given an image of size widthxheight in srcImg, smooths the image using a gaussian filter (cvSmooth) and copies the smoothed image to smoothImg
/// If sigma=0.0, then calls cvSmooth(srcImg, smoothedImg, CV_GAUSSIAN, 5, 5); This is the default.
/// If sigma>0.0, then calls cvSmooth(srcImg, smoothedImg, CV_GAUSSIAN, 0, 0, sigma);
///
void SmoothImage(cv::Mat srcImg, unsigned char *smoothImg, double sigma)
{
  if (sigma <= 0)
  {
    CopyFromIplBuffer(smoothImg, srcImg);
    return;
  } //end-if

  cv::Mat img2{srcImg.cols, srcImg.rows, CV_8UC1, smoothImg};

  if (sigma == 1.0)
    cv::GaussianBlur(srcImg, img2, cv::Size(5,5), 1.25);
  else
    cv::GaussianBlur(srcImg, img2, cv::Size(0,0), sigma);
} //end-SmoothImage

///------------------------------------------------------------------------------------------
/// Perform Gauss filter on "src" and store the result in "dst"
///
static void GaussFilter(unsigned char *src, unsigned char *dst, int width, int height){
#if 0
  for (int j=0; j<width; j++){
    dst[j] = src[j];                    // row=0
    dst[width+j] = src[width+j];        // row=1

    dst[(height-2)*width+j] = src[(height-2)*width+j];  // row=height-2
    dst[(height-1)*width+j] = src[(height-1)*width+j];  // row=height-1
  } //end-for

  for (int i=2; i<height-2; i++){
    dst[i*width] = src[i*width];        // column=0
    dst[i*width+1] = src[i*width+1];    // column=1

    dst[i*width+width-2] = src[i*width+width-2];  // column=width-2
    dst[i*width+width-1] = src[i*width+width-1];  // column=width-1
  } //end-for
#else
  memcpy(dst, src, width*height);
#endif

#if 0
  // 5x5 kernel with sigma=1.0
  // { 1, 4, 7, 4, 1}
  // { 4, 16, 26, 16, 4}
  // { 7, 26, 41, 26, 7}
  // { 4, 16, 26, 16, 4}
  // { 1, 4, 7, 4, 1}
  for (int i=2; i<height-2; i++){
    for (int j=2; j<width-2; j++){
      dst[i*width+j] = 
        (src[(i-2)*width+j-2] + 4*src[(i-2)*width+j-1] + 7*src[(i-2)*width+j] + 4*src[(i-2)*width+j+1] + src[(i-2)*width+j+2] +
         4*src[(i-1)*width+j-2] + 16*src[(i-1)*width+j-1] + 26*src[(i-1)*width+j] + 16*src[(i-1)*width+j+1] + 4*src[(i-1)*width+j+2] +
         7*src[i*width+j-2] + 26*src[i*width+j-1] + 41*src[i*width+j] + 26*src[i*width+j+1] + 7*src[i*width+j+2] +
         4*src[(i+1)*width+j-2] + 16*src[(i+1)*width+j-1] + 26*src[(i+1)*width+j] + 16*src[(i+1)*width+j+1] + 4*src[(i+1)*width+j+2] +
         src[(i+2)*width+j-2] + 4*src[(i+2)*width+j-1] + 7*src[(i+2)*width+j] + 4*src[(i+2)*width+j+1] + src[(i+2)*width+j+2] + 137)/273;      
    } //end-for
  } //end-for

#else
  // Another 5x5 kernel with sigma=?
  //  {2, 4, 5, 4, 2},
  //  {4, 9, 12, 9, 4},
  //  {5, 12, 15, 12, 5},
  //  {4, 9, 12, 9, 4},
  //  {2, 4, 5, 4, 2}
  for (int i=2; i<height-2; i++){
    for (int j=2; j<width-2; j++){
      dst[i*width+j] =  
        (2*src[(i-2)*width+j-2] + 4*src[(i-2)*width+j-1] + 5*src[(i-2)*width+j] + 4*src[(i-2)*width+j+1] + 2*src[(i-2)*width+j+2] +
         4*src[(i-1)*width+j-2] + 9*src[(i-1)*width+j-1] + 12*src[(i-1)*width+j] + 9*src[(i-1)*width+j+1] + 4*src[(i-1)*width+j+2] +
         5*src[i*width+j-2] + 12*src[i*width+j-1] + 15*src[i*width+j] + 12*src[i*width+j+1] + 5*src[i*width+j+2] +
         4*src[(i+1)*width+j-2] + 9*src[(i+1)*width+j-1] + 12*src[(i+1)*width+j] + 9*src[(i+1)*width+j+1] + 4*src[(i+1)*width+j+2] +
         2*src[(i+2)*width+j-2] + 4*src[(i+2)*width+j-1] + 5*src[(i+2)*width+j] + 4*src[(i+2)*width+j+1] + 2*src[(i+2)*width+j+2] + 80) / 159;
    } //end-for
  } //end-for
#endif
} //end-GaussFilter


