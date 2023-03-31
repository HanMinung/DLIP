#include "myFunc.h"

Mat src, dst, src_gray, blurred, dst_morph;

// Global variables for Threshold
int threshold_value = 0;
int threshold_type = 0;
int Cval = 0;
int morphology_type = 0;

int const max_value = 255;
int const max_type = 6;
int const max_C = 10;
int const max_binary_value = 255;


// Global variables for Morphology
int element_shape = MORPH_RECT;
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

// Trackbar strings
String window_name = "Threshold & Morphology Demo";
String trackbar_type = "Thresh Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Invertd \n 5: Otsu method \n 6: Adaptive threshold";
String Cvalue = "C in adaptive filter";
String trackbar_value = "Thresh Value";
String trackbar_morph = "Morph Type 0: None \n 1: erode \n 2: dilate \n 3: close  \n 4: open";


// Function declaration
void Threshold_Demo(int, void*);
void Morphology_Demo(int, void*);


void main(){

	int i = 3;
	Size kernelSize = cv::Size(i, i);

	src = imread("Lab_GrayScale_TestImage.jpg", IMREAD_COLOR);
	checkLoad(src);

	cvtColor(src, src_gray, CV_BGR2GRAY);

	cv::medianBlur(src_gray, blurred, 3);
	cv::GaussianBlur(src_gray, blurred, cv::Size(i, i), 0);
	//namedWindow("Bluured", WINDOW_FREERATIO);
	//imshow("Bluured", blurred);

	namedWindow(window_name, WINDOW_NORMAL);

		createTrackbar(trackbar_type, window_name, &threshold_type, max_type, Threshold_Demo);
		createTrackbar(trackbar_value, window_name, &threshold_value, max_value, Threshold_Demo);
		createTrackbar(Cvalue, window_name, &Cval, max_C, Threshold_Demo);
		createTrackbar(trackbar_morph, window_name, &morphology_type, max_type, Morphology_Demo);

		Threshold_Demo(0, 0);
		Morphology_Demo(0, 0);

	while (true) {

		int c = waitKey(20);
		if (c == 27)
			break;
	}

	waitKey(0);
}


void Threshold_Demo(int, void*) {

	// 0: Binary	1: Threshold Truncated	  2: Threshold to Zero	3: Threshold to Zero Inverted
	// 4: To zero inverted		5: Otsu method		6: adaptive threshold
	
	if (threshold_type == 5)  threshold_type = 8;

	threshold(blurred, dst, threshold_value, max_binary_value, threshold_type);

	if (threshold_type == 6)  adaptiveThreshold(blurred, dst, max_binary_value, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, Cval);

	imshow(window_name, dst);
}


// default form of callback function for trackbar
void Morphology_Demo(int, void*) {

	//	0: None	 1: Erode	2: Dilate	3: OPEN 	4: CLOSE
	
	switch (morphology_type) {

	case 0: dst.copyTo(dst_morph);									break;
	case 1: erode(dst, dst_morph, element);							break;
	case 2: dilate(dst, dst_morph, element);						break;
	case 3: morphologyEx(dst, dst_morph, CV_MOP_OPEN, element);		break;
	case 4: morphologyEx(dst, dst_morph, CV_MOP_CLOSE, element);	break;
	}

	imshow(window_name, dst_morph);
}
