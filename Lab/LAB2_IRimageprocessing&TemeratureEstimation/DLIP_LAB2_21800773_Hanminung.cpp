#include "myFunc.h"

Mat src, srcGray, srcHSV, dstBlur, dstMorph, dstHSV, dstConcat, hsvGray;
Mat graySort, maskResult, contourMask;

#define hmin					(int) 0
#define hmax					(int) 179
#define smin					(int) 0
#define smax					(int) 255
#define vmin					(int) 144
#define vmax					(int) 255

#define criticalMin				(int) 13000

#define videoWidth				(int) 640
#define videoHeight				(int) 480

Scalar minRange = Scalar(hmin, smin, vmin);
Scalar maxRange = Scalar(hmax, smax, vmax);

vector<Mat> hsvSplit;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

void preProcessing(void);


void main() {

	VideoCapture cap("IR_DEMO_cut.avi");
	
	cap.set(CAP_PROP_FRAME_WIDTH, videoWidth);
	cap.set(CAP_PROP_FRAME_HEIGHT, videoHeight);

	Mat faceMask = cv::Mat::zeros(videoWidth, videoHeight, CV_8UC1);

	if (!cap.isOpened())	printf("Video Loading Failed ...!");

	while (true) {

		cap >> src;

		preProcessing();

		findContours(dstMorph, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		processContour(src, contours, hsvGray, "area", criticalMin);

		printImg("Src video", src);

		if (waitKey(20) == 27)
			break;

	}

}


void preProcessing(void) {

	cvtColor(src, srcHSV, CV_BGR2HSV);

	// inRange ภ๛ฟ๋วั image : dstHSV
	inRange(srcHSV, minRange, maxRange, dstHSV);

	dilate(dstHSV, dstMorph, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 2);

	// Split to H,S,V
	split(srcHSV, hsvSplit);
	hsvSplit[2].copyTo(hsvGray);

}

