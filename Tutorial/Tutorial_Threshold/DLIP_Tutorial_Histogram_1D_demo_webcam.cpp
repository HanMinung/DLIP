#include "opencv.hpp"
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

void plotHist(Mat, string, int, int);

int main(int argc, char* argv[]) {

	int fps				= 60;
	int delay_waitKey	= 1000 / fps;
	int keyIn			= 0;

	Mat src, dst;

	// Open Video
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	while (1) 
	{
		// Read image from video
		bool bSuccess = cap.read(src);
		if (!bSuccess) {
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		// Convert RGB to gray
		cvtColor(src, src, CV_BGR2GRAY);
		equalizeHist(src, dst);

		keyIn = waitKey(delay_waitKey);
		if (keyIn == 27) { // ESC key
			cout << "esc key is pressed by user" << endl;
			break;
		}
		
		imshow("src", src);
		imshow("dst", dst);	
		plotHist(src, "histplot_src", src.cols, src.rows);
		plotHist(dst, "histplot_dst", dst.cols, dst.rows);
	}

	return 0;
}

void plotHist(Mat src, string plotname, int width, int height) {

	// Compute the histograms 
	Mat hist;

	// Establish the number of bins (for uchar Mat type)
	int histSize = 256;
	
	// Set the ranges (for uchar Mat type)
	float range[] = { 0, 255 };

	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	double min_val, max_val;
	cv::minMaxLoc(hist, &min_val, &max_val);

	Mat hist_normed = hist * height / max_val;			// set max_val to fit the highest height of the plot image
	
	float bin_w = (float)width / histSize;				// set horizontal axis interval value according to the horizontal size of the plot image
	
	Mat histImage(height, width, CV_8UC1, Scalar(0));	// create 8-b	it white image of width and height size to be used for plot
	
	for (int i = 0; i < histSize - 1; i++) {
		// function of drawing lines
		line(	histImage,	
				Point((int)(bin_w * i),			height - cvRound(hist_normed.at<float>(i, 0))),		// ! Coordinate Understanding !
				Point((int)(bin_w * (i + 1)),	height - cvRound(hist_normed.at<float>(i + 1, 0))),
				Scalar(255), 2, 8, 0);															
	}

	imshow(plotname, histImage);
}
