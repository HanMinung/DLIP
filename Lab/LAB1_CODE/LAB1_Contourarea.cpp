#include "myFunc.h"

Mat src, dst, src_gray, dst_blurred, dst_thresh, dst_morph, dst_contour;

int threshold_value = 112;
int threshold_type = 0;
int Cval = 0;
int morphology_type = 3;

int const max_value = 255;
int const max_type = 6;
int const max_C = 10;
int const max_binary_value = 255;

double area = 0;

Scalar color1(255, 74, 200);		// M5_RECT_NUT : PURPLE 
Scalar color2(200, 74, 255);		// M5_HEX_NUT : PINK
Scalar color3(255, 100, 0);			// M6_HEX_NUT : BLUE
Scalar color4(0, 255, 0);			// M5_BOLT : GREEN 
Scalar color5(0, 0, 255);			// M6_BOLT : RED

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
std::vector<std::vector<Point>> M6_BOLT;
std::vector<std::vector<Point>> M5_BOLT;
std::vector<std::vector<Point>> M5_HEX_NUT;
std::vector<std::vector<Point>> M6_HEX_NUT;
std::vector<std::vector<Point>> M5_RECT_NUT;

// Global variables for Morphology
int element_shape = MORPH_RECT;
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

// M6 BOLT , M5 BOLT , M6 HEX-NUT , M5 HEX-NUT , RECT_NUT classification with matchshape
vector<int> refShape = { 48, 44, 29, 24, 13 };
vector<double> similarity(refShape.size(), 0);

void main() {

	int i = 3;
	Size kernelSize = cv::Size(i, i);

	src = imread("Lab_GrayScale_TestImage.jpg", IMREAD_COLOR);
	checkLoad(src);

	// Preprocess 1 : gray scale - Blur - Threshold - Erode
	cvtColor(src, src_gray, CV_BGR2GRAY);

	cv::GaussianBlur(src_gray, dst_blurred, cv::Size(i, i), 0);

	threshold(dst_blurred, dst_thresh, threshold_value, max_binary_value, threshold_type);

	erode(dst_thresh, dst_morph, element);

	// Get Contour Information
	findContours(dst_morph, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);


	for (int i = 0; i < contours.size(); i++) {

		area = contourArea(contours[i]);

		if (area > 2000 && area < 9000) {

			Moments mu = moments(contours[i], false);
			Point centroid = Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 50);
			cv::drawContours(src, contours, i, cv::Scalar(0, 0, 255), 2);
			putText(src, to_string(int(area)), centroid, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
		}

	}

	for (int i = 0; i < contours.size(); i++){

		if (hierarchy[i][3] != -1 && hierarchy[hierarchy[i][3]][2] != -1){

			int sibling = hierarchy[i][0];
			drawContours(src, contours, sibling, Scalar(255, 0, 0), 2);
			
		}
	}

	namedWindow("Contoured image", WINDOW_FREERATIO);
	imshow("Contoured image", src);

	waitKey(0);
}