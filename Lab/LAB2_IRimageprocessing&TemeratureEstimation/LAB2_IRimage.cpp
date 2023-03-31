#include "myFunc.h"

Mat src, srcGray, srcHSV, dstBlur, dstMorph, dstHSV, dstConcat;

Scalar minRange = Scalar(20, 20, 20);
Scalar maxRange = Scalar(45, 255, 255);

void main() {

	VideoCapture cap("IR_DEMO_cut.avi");

	// Check loading of video
	if (!cap.isOpened())	printf("Video Loading Failed ...!");

	while (true) {

		cap >> src;

		// Check loading of image 
		checkLoad(src);
		
		// Convert to grayscale
		cvtColor(src, srcHSV, CV_BGR2HSV);

		// Apply inrange function to source image
		inRange(srcHSV, minRange, maxRange, dstHSV);

		// Show src image
		namedWindow("Org", WINDOW_FREERATIO);
		imshow("Org", src);
		
		namedWindow("Inrange filtered Video", WINDOW_FREERATIO);
		imshow("Inrange filtered Video", dstHSV);

		if (waitKey(1) == 27)
			break;

	}
	
}