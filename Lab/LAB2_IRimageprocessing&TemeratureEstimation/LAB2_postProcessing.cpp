#include "myFunc.h"

Mat src, srcGray, srcHSV, srcROI, dstBlur, dstMorph, dstHSV, dstConcat, hsvGray, prtGray;
Mat graySort, maskResult, orgMask;

#define hmin	(int) 0
#define hmax	(int) 179
#define smin	(int) 0
#define smax	(int) 255
#define vmin	(int) 144
#define vmax	(int) 255

Scalar minRange = Scalar(hmin, smin, vmin);
Scalar maxRange = Scalar(hmax, smax, vmax);

vector<Mat> hsvSplit;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

int cnt = 0;
float sumGray = 0, maxGray = 0, avgGray = 0;
bool warnFlag = 0;

void main() {

	src = imread("person_1.jpg", IMREAD_COLOR);

	resize(src, src, Size(640, 480));

	Mat faceMask = Mat::zeros(src.size(), CV_8UC1);
	Mat contourMask = Mat::zeros(src.size(), CV_8UC1);

	// setting for ROI selection
	cv::Size srcSize = src.size();
	std::cout << "Image size : " << srcSize.width << " x " << srcSize.height << "\n" << std::endl;

	// Check loading of image 
	checkLoad(src);

	cvtColor(src, srcHSV, CV_BGR2HSV);

	inRange(srcHSV, minRange, maxRange, dstHSV);

	dilate(dstHSV, dstMorph, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 2);

	// Split matrix to get gray scale temperature : 색상, 채도, 명도
	split(srcHSV, hsvSplit);
	hsvSplit[2].copyTo(hsvGray);
	 
	findContours(dstMorph, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (int idx = 0; idx < contours.size(); idx++) {

		int area = contourArea(contours[idx]);

		if (area > 14000) {

			Moments mu = moments(contours[idx], false);
			Point centroid = Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 50);
			cv::drawContours(src, contours, idx, Scalar(0, 0, 255), 2);
			
			drawContours(contourMask, contours, idx, Scalar(255), FILLED);

			bitwise_or(faceMask, contourMask, faceMask);
			
		}
	}

	bitwise_and(faceMask, hsvGray, maskResult);

	hsvGray.copyTo(prtGray);
	maskResult.copyTo(orgMask);

	graySort = maskResult.reshape(0, 1);
	cv::sort(graySort, graySort, SORT_DESCENDING);
	
	for (int idx = 0; idx < graySort.cols * 0.05; idx ++){

		// To access specific pixel : source.at<uchar>(x,y)
		if (graySort.at<uchar>(0, idx) != 0){

			if (maxGray <= graySort.at<uchar>(0, idx))	maxGray = graySort.at<uchar>(0, idx);
				
			sumGray += graySort.at<uchar>(0, idx);

			cnt ++;
		}
	}

	if (cnt != 0)	avgGray = (float)(sumGray / (float)cnt);
	
	printf("max temp : %lf\t\t avg temp : %lf\n", PIX2DEG(graySort.at<uchar>(0, 0)), PIX2DEG(avgGray));

	// Results
	printImg("Org", src);
	printImg("Inrange + Dilation result", dstMorph);
	printImg("HSV gray image", prtGray);
	printImg("Masking image", faceMask);
	printImg("Masked result", orgMask);

	waitKey(0);

}
