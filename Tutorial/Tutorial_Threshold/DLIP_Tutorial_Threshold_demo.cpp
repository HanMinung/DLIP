/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Threshold Demo
* Created: 2023-Spring
------------------------------------------------------*/
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//* @function main
int main(){

	Mat src, src_gray, dst, dst_morph;
	
	src = imread("../../Image/Thresholding&Morphology/coin.jpg", 0);    

	if (src.empty())							
	{
		cout << "File Read Failed : src is empty" << endl;
		waitKey(0);
	}

	namedWindow("DemoWindow", CV_WINDOW_AUTOSIZE);	//CV_WINDOW_AUTOSIZE(1) :Fixed Window, 0: Unfixed window
	if (!src.empty()) imshow("DemoWindow", src);    // Show image

	/* 
	[threshold_type]
	0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
	*/

	int threshold_value = 125;
	int threshold_type	= 0;
	int const max_value = 255;
	int const max_type	= 4;
	int const max_binary_value = 255;

	threshold(src, dst, threshold_value, max_binary_value, threshold_type);

	// Create a window to display results
	namedWindow("ThreshWIndow", CV_WINDOW_AUTOSIZE); //CV_WINDOW_AUTOSIZE(1) :Fixed Window, 0: Unfixed window
	imshow("ThreshWIndow", dst); // Show image

	waitKey(0);//Pause the program
	return 0;
}
