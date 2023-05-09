/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV Tutorial: Flip exercise (Image based)
* Created: 2021-Spring
------------------------------------------------------*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main()
{
	/*  open the video camera no.0  */
	VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);

	int process_type = 0;

	while (1)
	{
		Mat frame, dst;

		/*  read a new frame from video  */
		bool bSuccess = cap.read(frame);

		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}

		int key = waitKey(5);

		if (key == 'h') // wait for 'ESC' press for 30ms. If 'ESC' is pressed, break loop
		{
			process_type = 1;
		}
		else if (key == 'v')
		{
			process_type = 2;
		}
		else if (key == 'o')
		{
			process_type = 0;
		}

		if (process_type == 0)     dst = frame;
		else if (process_type == 1) flip(frame, dst, 0);
		else                       flip(frame, dst, 1);


		imshow("MyVideo", dst);
	}
}