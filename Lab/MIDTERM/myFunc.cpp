#pragma once

#include "myFunc.h"

// Function Definition File

void checkSize(Mat src) {

	Size imgSize = src.size();
	
	printf("Image width : %d \nImage height : %d", imgSize.width, imgSize.height);
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


void drawPoint(Mat& src, Point point, string colorName) {

	Scalar color(0, 0, 0);

	if (colorName == "red")		color = Scalar(0, 0, 255);
	if (colorName == "blue")	color = Scalar(255, 0, 0);
	if (colorName == "green")	color = Scalar(0, 255, 0);
	if (colorName == "white")	color = Scalar(255, 255, 255);
	if (colorName == "black")	color = Scalar(0, 0, 0);
	
	circle(src, point, 3, color, 2);
}


void localFilter(Mat& src, string filterType, Point point, int roiWidth, int roiHeight, string colorName) {

	Scalar color(0, 0, 0);

	if (colorName == "red")		color = Scalar(0, 0, 255);
	if (colorName == "blue")	color = Scalar(255, 0, 0);
	if (colorName == "green")	color = Scalar(0, 255, 0);
	if (colorName == "white")	color = Scalar(255, 255, 255);
	if (colorName == "black")	color = Scalar(0, 0, 0);

	Mat roi = src(Rect(point.x, point.y, roiWidth, roiHeight));
	Mat filteredRoi;

	int size = 3;

	if (filterType == "box")			blur(roi, filteredRoi, cv::Size(size, size), cv::Point(-1, -1));

	if (filterType == "gaussian")		GaussianBlur(roi, filteredRoi, cv::Size(3, 3), 0, 0);

	if (filterType == "median")			medianBlur(roi, filteredRoi, 3);
	
	if (filterType == "laplacian")		laplacianFilter(roi, filteredRoi);
	
	filteredRoi.copyTo(src(Rect(point.x, point.y, roiWidth, roiWidth)));
	rectangle(src, Rect(point.x, point.y, roiWidth, roiHeight), color, 2);

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

	src.convertTo(src, CV_8UC3);
	equalizeHist(src, dst);
}


// ---------------------------------------------------------------
// Function to blur a source image to reduce the effect of noise
// mode : box filter | gaussian filter | median filter
void blurImg(Mat& src, Mat& dst, string mode, int kernelSize) {

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

	//	[ threshold type ]
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


void roiImg(Mat& src, Mat& srcRoi, int roiX, int roiY, int roiWidth, int roiHeight) {

	Rect roi(roiX, roiY, roiWidth, roiHeight);
	
	src(roi).copyTo(srcRoi);
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
// mode : Area | length
// criticalMin : threshold value to process contour
// vector<vector<Point>> contours;
// vector<Vec4i> hierarchy;
void processContour(Mat& src, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin, string colorName) {

	int cnt = 0;
	Scalar color(0, 0, 0);

	if (colorName == "red")		color = Scalar(0, 0, 255);
	if (colorName == "blue")	color = Scalar(255, 0, 0);
	if (colorName == "green")	color = Scalar(0, 255, 0);
	if (colorName == "white")	color = Scalar(255, 255, 255);
	if (colorName == "black")	color = Scalar(0, 0, 0);

	if (mode == "area") {

		for (int Idx = 0; Idx < contours.size(); Idx++) {

			int area = cv::contourArea(contours[Idx]);
			
			if (area > criticalMin) {

				Rect BoundingBox = cv::boundingRect(contours[Idx]);
				Moments mu = cv::moments(contours[Idx], false);
				Point centroid = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
				//drawContours(src, contours, Idx, cv::Scalar(0, 0, 255), 2);
				rectangle(src, BoundingBox, color, 2);
				putText(src, to_string(int(area)), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);

				cnt++; 

			}
		}
		
		putText(src, "# of objects : " + to_string(cnt), Point(5, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 50), 3);
	}

	if (mode == "length") {

		for (int Idx = 0; Idx < contours.size(); Idx++) {

			int Len = cv::arcLength(contours[Idx], true);

			if (Len > criticalMin) {

				Rect BoundingBox = cv::boundingRect(contours[Idx]);
				rectangle(src, BoundingBox, color, 2);
				Moments mu = cv::moments(contours[Idx], false);
				Point centroid = cv::Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 50);
				drawContours(src, contours, Idx, cv::Scalar(0, 0, 255), 2);
				//putText(src, to_string(int(Len)), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
			}
		}
	}

}


// Process contour Version 2
// 사각형의 비율을 생각하여 조건을 추가한 버전
void processContourRatio(Mat& src, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin, string colorName) {

	Scalar color(0, 0, 0);
	int cnt = 0;

	if (colorName == "red")		color = Scalar(0, 0, 255);
	if (colorName == "blue")	color = Scalar(255, 0, 0);
	if (colorName == "green")	color = Scalar(0, 255, 0);
	if (colorName == "white")	color = Scalar(255, 255, 255);
	if (colorName == "black")	color = Scalar(0, 0, 0);

	printf("Box No.       Pixel(W x H)       Size(W[cm] * H[cm]) = Area[cm2]\n");

	Mat redROI;
	src.copyTo(redROI);

	if (mode == "area") {

		for (int Idx = 0; Idx < contours.size(); Idx++) {

			int area = cv::contourArea(contours[Idx]);

			if (area > criticalMin) {

				Rect BoundingBox = boundingRect(contours[Idx]);
				double ratio = static_cast<double>(BoundingBox.width) / static_cast<double>(BoundingBox.height);

				Moments mu = cv::moments(contours[Idx], false);
				Point centroid = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
				//drawContours(src, contours, Idx, cv::Scalar(0, 0, 255), 2);
				rectangle(src, BoundingBox, color, 2);

				putText(src, to_string(int(area)), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
				cnt++;

				Mat roi = src(Rect(BoundingBox.x, BoundingBox.y, BoundingBox.width, BoundingBox.height));
				
				//redRoi.copyTo(src(Rect(BoundingBox.x, BoundingBox.y, BoundingBox.width, BoundingBox.height)));
				rectangle(src, Rect(BoundingBox.x, BoundingBox.y, BoundingBox.width, BoundingBox.height), color, 2);
				
				rectangle(redROI, Rect(BoundingBox.x, BoundingBox.y, BoundingBox.width, BoundingBox.height), Scalar(0, 0, 255), CV_FILLED);
				
				printf("---------------------------------------------------------------------------\n");
				printf("Box [%d]       %d x %d             %.1f x %.1f = %.1f\n", cnt, int(BoundingBox.width), int(BoundingBox.height), (5.0 * BoundingBox.width)/72.5, (5.0 * BoundingBox.height) / 72.5, (5.0 * BoundingBox.width) / 72.5 * (5.0 * BoundingBox.height) / 72.5);

			}
		}
		
		printImg("redROI", redROI);
		putText(src, "# of objects : " + to_string(cnt), Point(5, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 50), 3);
	}
	
	if (mode == "length") {

		for (int Idx = 0; Idx < contours.size(); Idx++) {

			int Len = cv::arcLength(contours[Idx], true);

			if (Len > criticalMin) {

				cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
				double ratio = static_cast<double>(BoundingBox.width) / static_cast<double>(BoundingBox.height);

				if (ratio <= 1.0) { 

					rectangle(src, BoundingBox, cv::Scalar(0, 0, 255), 2);
					Moments mu = cv::moments(contours[Idx], false);
					Point centroid = cv::Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 50);
					drawContours(src, contours, Idx, color, 2);
					putText(src, to_string(int(Len)), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
				}
			}
		}
	}
}


// Version for counting the number of target
//int processContourRatioNum(Mat& src, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin, string colorName) {
//
//	Scalar color(0, 0, 0);
//	int cnt = 0;
//
//	if (colorName == "red")		color = Scalar(0, 0, 255);
//	if (colorName == "blue")	color = Scalar(255, 0, 0);
//	if (colorName == "green")	color = Scalar(0, 255, 0);
//	if (colorName == "white")	color = Scalar(255, 255, 255);
//	if (colorName == "black")	color = Scalar(0, 0, 0);
//
//	if (mode == "area") {
//
//		for (int Idx = 0; Idx < contours.size(); Idx++) {
//
//			int area = cv::contourArea(contours[Idx]);
//
//			if (area > criticalMin) {
//
//				Rect BoundingBox = boundingRect(contours[Idx]);
//				double ratio = static_cast<double>(BoundingBox.width) / static_cast<double>(BoundingBox.height);
//
//				if (ratio <= 1.0) { // width/height 비율이 1 이하인 경우에만 사각형 그리기
//
//					Moments mu = cv::moments(contours[Idx], false);
//					Point centroid = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
//					//drawContours(src, contours, Idx, cv::Scalar(0, 0, 255), 2);
//					rectangle(src, BoundingBox, color, 2);
//
//					putText(src, to_string(int(area)), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 3);
//
//					cnt ++;
//
//					// Bounding box에 해당하는 부분에 대해 roi를 설정
//					Mat roi = src(Rect(BoundingBox.x, BoundingBox.y, BoundingBox.width, BoundingBox.height));
//					Mat filteredRoi;
//
//					int size = 3;
//
//					// roi에 대해 gaussian blur를 적용하고
//					blurImg(roi, filteredRoi, "gaussian", 3);
//
//					// 모자이크가 됐다면 해당 부분을 원본 이미지에 복사
//					filteredRoi.copyTo(src(Rect(BoundingBox.x, BoundingBox.y, BoundingBox.width, BoundingBox.height)));
//
//					// bounding box에 해당하는 부분에 사각형을 쳐주는 부분
//					cv::rectangle(src, Rect(BoundingBox.x, BoundingBox.y, BoundingBox.width, BoundingBox.height), color, 2);
//
//					// ---------------------------------------------------------------------------------
//
//				}
//			}
//		}
//	}
//
//	if (mode == "length") {
//
//		for (int Idx = 0; Idx < contours.size(); Idx++) {
//
//			int Len = cv::arcLength(contours[Idx], true);
//
//			if (Len > criticalMin) {
//
//				cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
//				double ratio = static_cast<double>(BoundingBox.width) / static_cast<double>(BoundingBox.height);
//
//				if (ratio <= 1.0) {
//
//					rectangle(src, BoundingBox, cv::Scalar(0, 0, 255), 2);
//					Moments mu = cv::moments(contours[Idx], false);
//					Point centroid = cv::Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 50);
//					drawContours(src, contours, Idx, color, 2);
//					putText(src, to_string(int(Len)), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
//
//					cnt++;
//				}
//			}
//		}
//	}
//	
//	return cnt;
//}



// Function for image processing
// Input parameter : hmin, hmax, smin, smax, vmin, vmax
void inrangeImg(Mat& src, Mat& dstHSV, int hmin, int smin, int vmin, int hmax, int smax, int vmax) {

	Scalar rangeMin = Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax));
	Scalar rangeMax = Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax));

	inRange(src, rangeMin, rangeMax, dstHSV);
}


// Function to find circle in a image
// int const maxDist = 400;		 인접한 원들 사이의 최소 거리
// int const maxRadius = 400;	 검출할 원의 최소반지름
// input Image : gray scale + filtered image
void circleDet(Mat& src,Mat& srcGray , int minDist, int minRadius) {

	vector<Vec3f> circles;

	HoughCircles(srcGray, circles, HOUGH_GRADIENT, 2, minDist, 200, 100, minRadius, 500);

	for (size_t i = 0; i < circles.size(); i++) {

		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		// Center of detected circle
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);

		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);

	}
}


// Function for edge detection
// input : grayscale image + blurred image
void edgeDet(const Mat& src, Mat& result, double lowThreshold, int kernelSize){

	Canny(src, result, lowThreshold, 3 * lowThreshold, kernelSize, false);

	//printImg("Edges", result);
}


// Function for line detection
// Input image : Canny edge detection을 완료한 이미지
void problineDet(Mat& src, Mat& dst, int threshold, int minDist) {

	vector<Vec4i> linesP;
	HoughLinesP(src, linesP, 1, CV_PI / 180, threshold, minDist, 10);

	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++) {

		Vec4i l = linesP[i];
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
}



//void processContour(cv::Mat& image, std::vector<std::vector<cv::Point>>& contours, cv::Mat& hsvV, string mode, int criticalMin) {
//
//	Mat contourMask = Mat::zeros(image.size(), CV_8UC1);
//	Mat faceMask = Mat::zeros(image.size(), CV_8UC1);
//	Mat maskResult = Mat::zeros(image.size(), CV_8UC1);
//	Mat prtGray = Mat::zeros(image.size(), CV_8UC1);
//	Mat graySort;
//
//	float maxGray = 0.0, sumGray = 0.0, avgGray = 0.0;
//	int cnt = 0;
//
//	if (mode == "area") {
//
//		int maxArea = 0;
//		int maxIdx = 0;
//
//		for (int Idx = 0; Idx < contours.size(); Idx++) {
//
//			int area = cv::contourArea(contours[Idx]);
//
//			if (maxArea < area) {
//
//				maxArea = area;
//				maxIdx = Idx;
//			}
//		}
//
//		if (maxArea > criticalMin) {
//
//			cv::Rect BoundingBox = cv::boundingRect(contours[maxIdx]);
//			cv::Moments mu = cv::moments(contours[maxIdx], false);
//			cv::Point centroid = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
//			cv::drawContours(image, contours, maxIdx, cv::Scalar(0, 0, 255), 2);
//			cv::rectangle(image, BoundingBox, cv::Scalar(0, 0, 255), 2);
//
//			drawContours(contourMask, contours, maxIdx, Scalar(255), FILLED);
//			bitwise_or(faceMask, contourMask, faceMask);
//
//			bitwise_and(faceMask, hsvV, maskResult);
//
//			hsvV.copyTo(prtGray);
//
//			graySort = maskResult.reshape(0, 1);
//			cv::sort(graySort, graySort, SORT_DESCENDING);
//
//			for (int idx = 0; idx < graySort.cols * 0.01; idx++) {
//
//				if (graySort.at<uchar>(0, idx) != 0) {
//
//					if (maxGray <= graySort.at<uchar>(0, idx))	maxGray = graySort.at<uchar>(0, idx);
//
//					sumGray += graySort.at<uchar>(0, idx);
//
//					cnt++;
//				}
//			}
//
//			if (cnt != 0)	avgGray = (float)(sumGray / (float)cnt);
//
//			putText(image, "Max temp : " + to_string(int(PIX2DEG(graySort.at<uchar>(0, 0)))) + "  AVG temp : " + to_string(int(PIX2DEG(avgGray))), Point(5, 25), FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 0, 255));
//
//			if (PIX2DEG(avgGray) >= 38.0)	putText(image, "WARNING !", Point(5, 80), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 0, 255));
//		}
//	}
//
//
//	if (mode == "length") {
//
//		for (int Idx = 0; Idx < contours.size(); Idx++) {
//
//			int area = cv::arcLength(contours[Idx], true);
//
//			if (area > criticalMin) {
//
//				cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
//				cv::rectangle(image, BoundingBox, cv::Scalar(0, 0, 255), 2);
//				cv::Moments mu = cv::moments(contours[Idx], false);
//				cv::Point centroid = cv::Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 50);
//				cv::drawContours(image, contours, Idx, cv::Scalar(0, 0, 255), 2);
//			}
//		}
//	}
//}