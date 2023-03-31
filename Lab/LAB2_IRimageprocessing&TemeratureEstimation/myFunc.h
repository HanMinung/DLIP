#pragma once
#define _CRT_SECURE_NO_WARNGINS

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void printImg(string str, Mat src);

void checkLoad(Mat src);

void plotHist(Mat src, string plotname, int width, int height);

void processContour(cv::Mat& image, std::vector<std::vector<cv::Point>>& contours, cv::Mat& hsvV, string mode, int criticalMin);

float PIX2DEG(float val);