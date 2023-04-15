#pragma once
#define _CRT_SECURE_NO_WARNGINS

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

void checkSize(Mat src);

void resizeImg(Mat& src, int imgWidth, int imgHeight);

void checkSize(Mat src);

void printImg(string str, Mat src);

void drawPoint(Mat& src, Point point, string color);

void localFilter(Mat& src, string filterType, Point point, int roiWidth, int roiHeight, string colorName);

void checkLoad(Mat src);

void plotHist(Mat src, string plotname, int width, int height);

void histogrmaEqual(Mat& src, Mat& dst);

void roiImg(Mat& src, Mat& srcRoi, int roiX, int roiY, int roiWidth, int roiHeight);

// Filter function with mode selection  ----------------------------------
void blurImg(Mat& src, Mat& dst, string mode, int kernelSize);

void laplacianFilter(Mat& src, Mat& result);

void thresholdImg(Mat& src, Mat& dst, int criticalVal);

void morphImg(Mat& src, Mat& dst, string type, int iteration);

void processContour(cv::Mat& image, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin, string colorName);

void processContourRatio(Mat& src, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin, string colorName);

//int processContourRatioNum(Mat& src, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin, string colorName);

// Proper range for HSV color segmentation
// Yello : (20, 100, 100) ~ (40, 255, 255)
// Red   : (0, 100, 100) ~ (10, 255, 255) or (170, 100, 100) ~ (180, 255, 255)
// Blue  : (121, 180, 76) ~ (90, 255, 255)
// Face : (0, 55, 145), (15, 185, 255)
void inrangeImg(Mat& src, Mat& dstHSV, int hmin, int smin, int vmin, int hmax, int smax, int vmax);

void circleDet(Mat& src, Mat& srcGray, int minDist, int minRadius);

void edgeDet(const Mat& src, Mat& output, double lowThreshold, int kernelSize);

void problineDet(Mat& src, Mat& dst, int threshold, int minDist);

  