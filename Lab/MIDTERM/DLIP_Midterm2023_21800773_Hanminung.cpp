
/* 

	DLIP MIDTERM 2023-1  Submission

	NAME:Hanminung

*/

#include "../../Include/myFunc.h"
// 제출용 경로 : 
// #include "Include/myFunc.h"

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

int thresholdVal = 150;
int const minArea = 4000;
int const maxVal = 255;

String windowName = "Threshold & Morphology Demo";
String thresholdValue = "Thresh Value";

Mat src, srcGray, srcThresh, srcBlur, srcEqual, srcMorph, dstSegment, dstContour;

void Threshold_Demo(int, void*);

int main(){
	
	src = imread("midterm_brick1.jpg");				// test image

	checkLoad(src);									// Check the source properly loaded

	printImg("src", src);

	cvtColor(src, srcGray, CV_BGR2GRAY);

	blurImg(srcGray, srcBlur, "gaussian", 5);
	printImg("Gaussuan blur", srcBlur);
	
	namedWindow(windowName, WINDOW_NORMAL);
	createTrackbar(thresholdValue, windowName, &thresholdVal, maxVal, Threshold_Demo);

	Threshold_Demo(0, 0);

	/**** Segmentation of Objects (bricks and box) ****/
	srcMorph.copyTo(dstSegment);

	/**** Count Number of Objects and Calculate Area of Objects ****/
	findContours(dstSegment, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	
	src.copyTo(dstContour);
	
	processContourRatio(dstContour, contours, "area", minArea, "green");
	printImg("contour result", dstContour);
	
	/**** Fill Segmented Object as Red color ****/
	// This code is included in function 'processContourRatio'

	
	waitKey(0);
	return 0;
}


void Threshold_Demo(int, void*) {

	threshold(srcBlur, srcThresh, thresholdVal, 255, 0);
	//printImg("threhold result", srcThresh);
	printImg("Thresholding", srcThresh);

	morphImg(srcThresh, srcMorph, "dilate", 4);
	morphImg(srcMorph, srcMorph, "erode", 3);

	printImg(windowName, srcMorph);
}