/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Threshold Demo
* Created: 2023-Spring
------------------------------------------------------*/
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void plotHist(Mat src, string plotname, int width, int height);

int flagImg = 1;

int main()
{
	Mat src, dst, combined;

	if(flagImg == 1)		src = imread("../../Image/Thresholding&Morphology/localThresh1.jpg", 0);   
	else if(flagImg == 2)	src = imread("../../Image/Thresholding&Morphology/localThresh2.jpg", 0);   
	else					src = imread("../../Image/Thresholding&Morphology/localThresh3.jpg", 0);   

	if (src.empty()){

		cout << "File Read Failed : src is empty" << endl;
		waitKey(0);
	}

	int threshold_value = 130;
	int const max_binary_value	= 255;

	if (flagImg == 1) {

		adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 11);
		Size size = src.size();

		//namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
		//imshow("Original", src);

		//plotHist(src, "Histogrma of src image", size.width, size.height);

		//namedWindow("LOCAL THRESHOLD 1", CV_WINDOW_AUTOSIZE); 
		//imshow("LOCAL THRESHOLD 1", dst);

		hconcat(src, dst, combined);
		namedWindow("Before and After", CV_WINDOW_AUTOSIZE);
		imshow("Before and After", combined);
		waitKey(0);

	}

	else if(flagImg == 2) {

		adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 3);
		
		namedWindow("LOCAL THRESHOLD 2", CV_WINDOW_AUTOSIZE);
		imshow("LOCAL THRESHOLD 2", dst);
	}

	else {

		adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 1);

		namedWindow("LOCAL THRESHOLD 3", CV_WINDOW_AUTOSIZE);
		imshow("LOCAL THRESHOLD 3", dst);
	}

	waitKey(0);
	return 0;
}


void plotHist(Mat src, string plotname, int width, int height) {
	/// Compute the histograms 
	Mat hist;
	
	/// Establish the number of bins (for uchar Mat type)
	int histSize = 256;
	
	/// Set the ranges (for uchar Mat type)
	float range[] = { 0, 256 };

	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	double min_val, max_val;
	cv::minMaxLoc(hist, &min_val, &max_val);
	
	Mat hist_normed = hist * height / max_val;
	float bin_w = (float)width / histSize;

	Mat histImage(height, width, CV_8UC1, Scalar(0));
	for (int i = 0; i < histSize - 1; i++) {
		line(	histImage,
				Point((int)(bin_w * i),			height - cvRound(hist_normed.at<float>(i, 0))),
				Point((int)(bin_w * (i + 1)),	height - cvRound(hist_normed.at<float>(i + 1, 0))),
				Scalar(255), 2, 8, 0);
	}

	imshow(plotname, histImage);
}