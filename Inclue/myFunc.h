#pragma once
#define _CRT_SECURE_NO_WARNGINS

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

void checkSize(Mat src);

void resizeImg(Mat& src, int imgWidth, int imgHeight);

void printImg(string str, Mat src);

void checkLoad(Mat src);

void plotHist(Mat src, string plotname, int width, int height);

void histogrmaEqual(Mat& src, Mat& dst);

// Filter function with mode selection  ----------------------------------
void blurImage(Mat& src, Mat& dst, string mode, int kernelSize);

void laplacianFilter(Mat& src, Mat& result);

void thresholdImg(Mat& src, Mat& dst, int criticalVal);

void morphImg(Mat& src, Mat& dst, string type, int iteration);


// Proper range for HSV color segmentation
// Yello : (20, 100, 100) ~ (30, 255, 255)
// Red   : (0, 100, 100) ~ (10, 255, 255) or (170, 100, 100) ~ (180, 255, 255)
// Blue  : (110, 100, 100) ~ (130, 255, 255)


void processContour(cv::Mat& image, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin);


