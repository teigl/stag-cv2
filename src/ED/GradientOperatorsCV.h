#ifndef GRADIENT_OPERATORS_CV_H
#define GRADIENT_OPERATORS_CV_H

#include <opencv2/core.hpp>

/// Compute color image gradient
void ComputeGradientMapByPrewitt(cv::Mat smoothImg, short *gradImg, unsigned char *dirImg, int GRADIENT_THRESH);

#endif