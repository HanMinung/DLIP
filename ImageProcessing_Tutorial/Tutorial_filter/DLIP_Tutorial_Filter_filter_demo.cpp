/* ------------------------------------------------------ /
*Image Proccessing with Deep Learning
* OpenCV : Filter Demo
* Created : 2021 - Spring
------------------------------------------------------ */

#include "../../Include/myFunc.h"

using namespace std;
using namespace cv;

void main()
{
	cv::Mat src, dst;
	src = cv::imread("../Image/blurry_moon.tif", 0);
	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("org", src);

	int i = 3;
	Size kernelSize = cv::Size(i, i);

	/* Blur */
	blurImage(src, dst, "box", 3);
	printImg("box filter", dst);

	/* Gaussian Filter */
	blurImage(src, dst, "gaussian", 3);
	printImg("gaussian filter", dst);

	/* Median Filter */
	blurImage(src, dst, "median", 3);
	printImg("median filter", dst);

	// Laplacian filter
	laplacianFilter(src, dst);
	printImg("laplacian filter",dst);


	cv::waitKey(0);
}