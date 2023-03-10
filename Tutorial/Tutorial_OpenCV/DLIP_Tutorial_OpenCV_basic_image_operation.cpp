/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV Tutorial: Basic Iamge Operation (Crop, Rotate, Resize)
* Created: 2021-Spring
------------------------------------------------------*/
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  read image  */
	Mat img = imread("../../Image/image.jpg");
	imshow("img", img);

	/*  Crop(Region of Interest)  */
	Rect r(200, 100, 50, 100);	 // (x, y, width, height)
	Mat roiImg = img(r);

	cout << "ROI IMAGE SIZE: " << roiImg.size() << endl;

	imshow("roiImg", roiImg);
	imwrite("roiImg.jpg", roiImg);

	/*  Rotate  */
	Mat rotImg;
	rotate(img, rotImg, ROTATE_90_CLOCKWISE);
	imshow("rotImg", rotImg);


	/*  Resize  */
	Mat resizedImg;
	resize(img, resizedImg, Size(1000, 100));
	imshow("resizedImg", resizedImg);

	waitKey(0);
}


