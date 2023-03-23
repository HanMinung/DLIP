#pragma once

#include "myFunc.h"

// Function Definition

void checkLoad(Mat src) {

	if (src.empty()) {

		cout << "File Read Failed : src is empty" << endl;
		waitKey(0);
	}
}


void plotHist(Mat src, string plotname, int width, int height) {

	Mat hist;

	int histSize = 256;

	float range[] = { 0, 256 };

	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	double min_val, max_val;
	cv::minMaxLoc(hist, &min_val, &max_val);

	Mat hist_normed = hist * height / max_val;
	float bin_w = (float)width / histSize;

	Mat histImage(height, width, CV_8UC1, Scalar(0));

	for (int i = 0; i < histSize - 1; i++) {

		line(histImage,
			Point((int)(bin_w * i), height - cvRound(hist_normed.at<float>(i, 0))),
			Point((int)(bin_w * (i + 1)), height - cvRound(hist_normed.at<float>(i + 1, 0))),
			Scalar(255), 2, 8, 0);
	}

	imshow(plotname, histImage);
}


