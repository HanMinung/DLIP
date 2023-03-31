#pragma once
#define _CRT_SECURE_NO_WARNGINS

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// Function Definition
void checkLoad(Mat src);

void plotHist(Mat src, string plotname, int width, int height);

