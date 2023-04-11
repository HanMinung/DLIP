/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Filter Demo - Video
* Created: 2021-Spring
------------------------------------------------------*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	int key = 0;
	int kernel_size = 3;
	int filter_type = 0;

	/*  open the video camera no.0  */
	VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);

	while (1){

		Mat src, dst;

		/*  read a new frame from video  */
		bool bSuccess = cap.read(src);

		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}

		key = waitKeyEx(30);

		if (key == 27) // wait for 'ESC' press for 30ms. If 'ESC' is pressed, break loop
		{
			cout << "ESC key is pressed by user\n";
			break;
		}
		else if (key == 'b' || key == 'B') {

			filter_type = 1;	// blur
		}

		else if (key == 'g' || key == 'G') {

			filter_type = 2;	// Gaussian Blur
		}

		else if (key == 'l' || key == 'L') {

			filter_type = 3;	// Laplacian filter
		}

		else if (key == 'r' || key == 'R') {

			filter_type = 4;	// Laplacian filter
		}

		else if (key == 'o' || key == 'O') {

			filter_type = 5;	// Laplacian filter
		}

		else if (key == 'u' || key == 'U') 
			kernel_size += 2;

		else if (key == 'd' || key == 'D') {
			
			if (kernel_size > 4)
				kernel_size -= 2;
		}



		if (filter_type == 1)
			blur(src, dst, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));

		else if (filter_type == 2)
			cv::GaussianBlur(src, dst, cv::Size(kernel_size, kernel_size), 0);

		else if (filter_type == 3) {

			cv::Laplacian(src, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
			src.convertTo(src, CV_16S);
			cv::Mat result_laplcaian = src - dst;
			result_laplcaian.convertTo(result_laplcaian, CV_8U);
			dst = result_laplcaian;

		}

		else
			src.copyTo(dst);
		
		imshow("MyVideo", dst);

	}

	return 0;
}