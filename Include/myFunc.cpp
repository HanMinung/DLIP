#pragma once

#include "myFunc.h"

// Function Definition File

void checkSize(Mat src) {

	Size imgSize = src.size();
	
	printf("Image width : %d, Image height : %d", imgSize.width, imgSize.height);
}


void resizeImg(Mat& src, int imgWidth, int imgHeight) {

	resize(src, src, Size(imgWidth, imgHeight));
}


void printImg(string str, Mat src) {

	namedWindow(str, WINDOW_FREERATIO);
	imshow(str, src);
}

// Check the success / failure of image loading
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

// Histogram equalization : apply after blurring
void histogrmaEqual(Mat& src, Mat& dst) {

	equalizeHist(src, dst);
}


// ---------------------------------------------------------------
// Function to blur a source image to reduce the effect of noise
// mode : box filter | gaussian filter | median filter
void blurImage(Mat& src, Mat& dst, string mode, int kernelSize) {

	int i = kernelSize;

	if (mode == "box")		blur(src, dst, cv::Size(i, i), cv::Point(-1, -1));

	if (mode == "gaussian") GaussianBlur(src, dst, cv::Size(i, i), 0);

	if (mode == "median")   medianBlur(src, dst, 3);
}


// ---------------------------------------------------------------
// Function to apply laplacian filter
// All process of applying that filter is in this code
void laplacianFilter(Mat& src, Mat& result) {

	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat dst;

	Laplacian(src, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
	src.convertTo(src, CV_16S);
	
	result = src - dst;
	result.convertTo(result, CV_8U);

}


void thresholdImg(Mat& src, Mat& dst, int criticalVal) {

	//	[threshold_type]
	//	0: Binary
	//	1: Binary Inverted
	//	2: Threshold Truncated
	//	3: Threshold to Zero
	//	4: Threshold to Zero Inverted

	int threshold_type = 0;
	int const max_value = 255;
	int const max_type = 4;
	int const max_binary_value = 255;

	threshold(src, dst, criticalVal, max_binary_value, threshold_type);

}



// Function for process morphology to an image
// Open  : erosion - dilation
// Close : dilation - erosion
void morphImg(Mat& src, Mat& dst, string type, int iteration) {

	// Global variables for Morphology
	int element_shape = MORPH_RECT;
	int n = 3;
	Mat element = getStructuringElement(element_shape, Size(n, n));

	if (type == "dilate")		dilate(src, dst, element, Point(-1, -1), iteration);
	if (type == "erode")		erode(src, dst, element, Point(-1, -1), iteration);
	if (type == "open")			morphologyEx(src, dst, CV_MOP_OPEN, element, Point(-1, -1), iteration);
	if (type == "close")		morphologyEx(src, dst, CV_MOP_CLOSE, element, Point(-1,-1), iteration);
	
}


// ---------------------------------------------------------------
//Function to process contours
//mode : Area | length
//criticalMin : threshold value to process contour
void processContour(cv::Mat& image, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin) {

	if (mode == "area") {

		for (int Idx = 0; Idx < contours.size(); Idx++) {

			int area = cv::contourArea(contours[Idx]);
			
			if (area > criticalMin) {

				cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
				cv::Moments mu = cv::moments(contours[Idx], false);
				cv::Point centroid = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
				cv::drawContours(image, contours, Idx, cv::Scalar(0, 0, 255), 2);
				cv::rectangle(image, BoundingBox, cv::Scalar(0, 0, 255), 2);
				printf("AREA : %d\n", area);
			}
		}
	}

	if (mode == "length") {

		for (int Idx = 0; Idx < contours.size(); Idx++) {

			int Len = cv::arcLength(contours[Idx], true);

			if (Len > criticalMin) {

				cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
				cv::rectangle(image, BoundingBox, cv::Scalar(0, 0, 255), 2);
				cv::Moments mu = cv::moments(contours[Idx], false);
				cv::Point centroid = cv::Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 50);
				cv::drawContours(image, contours, Idx, cv::Scalar(0, 0, 255), 2);
			}
		}
	}


}