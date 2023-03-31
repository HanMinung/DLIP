#pragma once

#include "myFunc.h"

// Function Definition File


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

// Function : processContour
// - This function takes in a contour as inputand processes it for later use.
// - The processing includes selecting the area OR length modes.
// - Depending on the threshold value, the function will perform additional post - processing steps.
void processContour(cv::Mat& image, std::vector<std::vector<cv::Point>>& contours , cv::Mat& hsvV,string mode, int criticalMin) {

	// FOR lab2
	Mat contourMask = Mat::zeros(image.size(), CV_8UC1);
	Mat faceMask = Mat::zeros(image.size(), CV_8UC1);
	Mat maskResult = Mat::zeros(image.size(), CV_8UC1);
	Mat prtGray = Mat::zeros(image.size(), CV_8UC1);
	Mat graySort;

	float maxGray = 0.0, sumGray = 0.0, avgGray = 0.0;
	int cnt = 0;

	if (mode == "area") {

		int maxArea = 0;
		int maxIdx = 0;

		for (int Idx = 0; Idx < contours.size(); Idx++) {

			int area = cv::contourArea(contours[Idx]);

			if (maxArea < area) {

				maxArea = area;
				maxIdx = Idx;
			}
		}

		if (maxArea > criticalMin) {

			cv::Rect BoundingBox = cv::boundingRect(contours[maxIdx]);
			cv::Moments mu = cv::moments(contours[maxIdx], false);
			cv::Point centroid = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
			cv::drawContours(image, contours, maxIdx, cv::Scalar(0, 0, 255), 2);
			cv::rectangle(image, BoundingBox, cv::Scalar(0, 0, 255), 2);

			// For lab2
			drawContours(contourMask, contours, maxIdx, Scalar(255), FILLED);
			bitwise_or(faceMask, contourMask, faceMask);

			bitwise_and(faceMask, hsvV, maskResult);

			//printImg("Masking result_1", maskResult);

			hsvV.copyTo(prtGray);

			graySort = maskResult.reshape(0, 1);
			cv::sort(graySort, graySort, SORT_DESCENDING);

			for (int idx = 0; idx < graySort.cols * 0.01; idx++) {

				// To access specific pixel : source.at<uchar>(x,y)
				if (graySort.at<uchar>(0, idx) != 0) {

					if (maxGray <= graySort.at<uchar>(0, idx))	maxGray = graySort.at<uchar>(0, idx);

					sumGray += graySort.at<uchar>(0, idx);

					cnt++;
				}
			}

			if (cnt != 0)	avgGray = (float)(sumGray / (float)cnt);
			
			putText(image, "Max temp : " + to_string(int(PIX2DEG(graySort.at<uchar>(0, 0)))) + "  AVG temp : " + to_string(int(PIX2DEG(avgGray))), 
				Point(5,25), FONT_HERSHEY_COMPLEX, 0.7, Scalar(0,0,255));
			
			if (PIX2DEG(avgGray) >= 38.0)	putText(image, "WARNING !", Point(5, 80), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 0, 255));
			//printf("max temp : %lf\t\t avg temp : %lf\n", PIX2DEG(graySort.at<uchar>(0, 0)), PIX2DEG(avgGray));
		}
	}


	if (mode == "length") {

		for (int Idx = 0; Idx < contours.size(); Idx++) {

			int area = cv::arcLength(contours[Idx], true);

			if (area > criticalMin) {

				cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
				cv::rectangle(image, BoundingBox, cv::Scalar(0, 0, 255), 2);
				cv::Moments mu = cv::moments(contours[Idx], false);
				cv::Point centroid = cv::Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 50);
				cv::drawContours(image, contours, Idx, cv::Scalar(0, 0, 255), 2);
			}
		}
	}
}


float PIX2DEG(float val) {

	return (float)round(15.0 * val / 255.0 + 25.0);
}








//void processContour(cv::Mat& image, std::vector<std::vector<cv::Point>>& contours, string mode, int criticalMin) {
//
//	if (mode == "area") {
//
//		for (int Idx = 0; Idx < contours.size(); Idx++) {
//
//			int area = cv::contourArea(contours[Idx]);
//			
//			if (area > criticalMin && area < criticalMax) {
//
//				cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
//				cv::Moments mu = cv::moments(contours[Idx], false);
//				cv::Point centroid = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
//				cv::drawContours(image, contours, Idx, cv::Scalar(0, 0, 255), 2);
//				cv::rectangle(image, BoundingBox, cv::Scalar(0, 0, 255), 2);
//				printf("AREA : %d\n", area);
//			}
//		}
//	}
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
//
//
//}