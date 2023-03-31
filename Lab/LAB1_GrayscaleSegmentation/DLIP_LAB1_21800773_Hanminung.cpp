#include "myFunc.h"

#define kernelSize				5

#define threshVal				118
#define maxBinaryVal			255
#define threshType				0

#define FORLOOP(i, Final)		for(int i = 0; i < Final ; i++)

#define M6BOLT					0
#define M5BOLT					1
#define M6HEXNUT				2
#define M5HEXNUT				3
#define M5RECTNUT				4 

int classType = 0;

Mat src, dst, srcGray, dstBlur, dstThresh, dstMorph, dstContour;
Mat preProcess;

Scalar color1(255, 74, 200);		// M5_RECT_NUT : PURPLE 
Scalar color2(200, 74, 255);		// M5_HEX_NUT : PINK
Scalar color3(255, 100, 0);			// M6_HEX_NUT : BLUE
Scalar color4(0, 255, 0);			// M5_BOLT : GREEN 
Scalar color5(0, 0, 255);			// M6_BOLT : RED

// Initialization of # of each component
int Bolt_M5 = 0;
int Bolt_M6 = 0;
int M6_HEXA_NUT = 0;
int M5_SQUARE_NUT = 0;
int M5_HEXA_NUT = 0;


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
Mat element = getStructuringElement(element_shape, Size(n, n));\


void preProcessing(Mat src);
void visualizing(int Idx,int Type);
int detType(int len);

void main() {

	src = imread("Lab_GrayScale_TestImage.jpg", IMREAD_COLOR);
	
	// Check image loading
	checkLoad(src);
	
	// image preprocessing
	preProcessing(src);

	//finding contour
	findContours(dstMorph, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	
	for (int i = 0; i < contours.size(); i++) {

		double len = arcLength(contours[i], true);
		
		Moments mu = moments(contours[i], false);
		Point centroid = Point(mu.m10 / mu.m00 + 30, mu.m01 / mu.m00 - 50);

		// Determine type
		classType = detType(len);

		// Drawing contours
		visualizing(i, classType);

	}

	int totalNum = Bolt_M5 + Bolt_M6 + M5_HEXA_NUT + M6_HEXA_NUT + M5_SQUARE_NUT;

	std::cout << "# of Total Components = "<< totalNum << "\n" << std::endl;
	std::cout << "# of M5_BOLT = " << Bolt_M5 << std::endl;
	std::cout << "# of M6_BOLT = " << Bolt_M6 << std::endl;
	std::cout << "# of M5_HEX_NUT = " << M5_HEXA_NUT << std::endl;
	std::cout << "# of M6_HEX_NUT = " << M6_HEXA_NUT << std::endl;
	std::cout << "# of M5_RECT_NUT = " << M5_SQUARE_NUT << std::endl;
	

	namedWindow("Contoured image", WINDOW_FREERATIO);
	imshow("Contoured image", src);

	waitKey(0);

}


void preProcessing(Mat src) {

	// gray scale
	cvtColor(src, srcGray, CV_BGR2GRAY);

	// Median blur with kernel size 5
	medianBlur(srcGray, dstBlur, kernelSize);

	// Threshold process
	threshold(dstBlur, dstThresh, threshVal, maxBinaryVal, threshType);

	// Morphology process
	dilate(dstThresh, dstMorph, element);

	dilate(dstMorph, dstMorph, element, Point(-1,-1), 13);
	erode(dstMorph, dstMorph, element, Point(-1,-1), 23);
	dilate(dstMorph, dstMorph, element, Point(-1, -1), 2);

}


int detType(int len) {

	if (len > 450)						classType = M6BOLT;
	if (len < 450 && len > 330)			classType = M5BOLT;
	if (len > 210 && len < 250)			classType = M6HEXNUT;
	if (len > 180 && len < 210)			classType = M5RECTNUT;
	if (len > 100 && len < 180)			classType = M5HEXNUT;

	return classType;
}


void visualizing(int Idx, int Type) {

	if (classType == M6BOLT) {

		cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
		cv::rectangle(src, BoundingBox, color1, 2);
		Moments mu = moments(contours[Idx], false);
		Point centroid = Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 80);
		putText(src, "M6 BOLT", centroid, FONT_HERSHEY_SIMPLEX, 1, color1, 2);

		Bolt_M6++;
	}


	if (classType == M5BOLT) {

		cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
		cv::rectangle(src, BoundingBox, color2, 2);
		Moments mu = moments(contours[Idx], false);
		Point centroid = Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 80);
		putText(src, "M5 BOLT", centroid, FONT_HERSHEY_SIMPLEX, 1, color2, 2);

		Bolt_M5++;
	}


	if (classType == M6HEXNUT) {

		cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
		cv::rectangle(src, BoundingBox, color3, 2);
		Moments mu = moments(contours[Idx], false);
		Point centroid = Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 80);
		putText(src, "M6 Hexa Nut", centroid, FONT_HERSHEY_SIMPLEX, 1, color3, 2);

		M6_HEXA_NUT++;
	}


	if (classType == M5RECTNUT) {

		cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
		cv::rectangle(src, BoundingBox, color4, 2);
		Moments mu = moments(contours[Idx], false);
		Point centroid = Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 80);
		putText(src, "M5 Rect Nut", centroid, FONT_HERSHEY_SIMPLEX, 1, color4, 2);

		M5_SQUARE_NUT++;
	}

	if (classType == M5HEXNUT) {

		cv::Rect BoundingBox = cv::boundingRect(contours[Idx]);
		cv::rectangle(src, BoundingBox, color5, 2);
		Moments mu = moments(contours[Idx], false);
		Point centroid = Point(mu.m10 / mu.m00 - 50, mu.m01 / mu.m00 - 80);
		putText(src, "M5 Hexa Nut", centroid, FONT_HERSHEY_SIMPLEX, 1, color5, 2);

		M5_HEXA_NUT++;
	}
}

