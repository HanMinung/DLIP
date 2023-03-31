#include "myFunc.h"

Mat src, dst, srcGray, dstBlur, dstThresh, dstMorph, dstContour, dstLapla;

int threshold_value = 115;
int threshold_type = 0;
int Cval = 0;

int const max_value = 255;
int const max_binary_value = 255;

// Initialization of # of each component
int Bolt_M5 = 0;
int Bolt_M6 = 0;
int M6_HEXA_NUT = 0;
int M5_HEXA_NUT = 0;
int M5_SQUARE_NUT = 0;

double length = 0;

vector<vector<Point>> contours;
std::vector<std::vector<Point>> M6_BOLT;
std::vector<std::vector<Point>> M5_BOLT;
std::vector<std::vector<Point>> M5_HEX_NUT;
std::vector<std::vector<Point>> M6_HEX_NUT;
std::vector<std::vector<Point>> M5_RECT_NUT;

// Global variables for Morphology
int element_shape = MORPH_RECT;
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

int scale = 1;
int delta = 0;
int ddepth = CV_16S;

void main(){

	int i = 5;
	Size kernelSize = cv::Size(i, i);

	src = imread("Lab_GrayScale_TestImage.jpg", IMREAD_COLOR);
	checkLoad(src);

	// Preprocess 1 : gray scale - Blur - Threshold - Erode
	cvtColor(src, srcGray, CV_BGR2GRAY);

	medianBlur(srcGray, dstBlur, 5);

	threshold(dstBlur, dstThresh, threshold_value, max_binary_value, threshold_type);
	
	Laplacian(dstThresh, dstLapla, ddepth, 9, scale, delta, cv::BORDER_DEFAULT);
	dstThresh.convertTo(dstThresh, CV_16S);
	//dstLapla = dstThresh - dstLapla;
	dstLapla.convertTo(dstLapla, CV_8U);

	//dilate(dstLapla, dstMorph, element, Point(-1, -1), 4);
	//erode(dstMorph, dstMorph, element, Point(-1, -1), 2);

	namedWindow("Contoured image", WINDOW_FREERATIO);
	imshow("Contoured image", dstLapla);

	waitKey(0);

}
