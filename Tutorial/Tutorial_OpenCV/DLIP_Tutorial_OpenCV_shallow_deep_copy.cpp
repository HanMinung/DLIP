#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{

	Mat src, dst_shallow, dst_deep;
	// read image  
	src = imread("../../Image/image.jpg", 0);

	/* Shallow Copy */
	dst_shallow = src;

	/* Deep Copy */
	src.copyTo(dst_deep);

	flip(src, src, 1);

	imshow("dst_shallow", dst_shallow);
	imshow("dst_deep", dst_deep);
	waitKey(0);
	return 0;
}
