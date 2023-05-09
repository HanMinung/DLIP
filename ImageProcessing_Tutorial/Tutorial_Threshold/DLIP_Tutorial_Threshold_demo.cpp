/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Threshold Demo
* Created: 2023-Spring
------------------------------------------------------*/

#include "../../Include/myFunc.h"


int thresholdVal = 100;

//* @function main
int main(){

	Mat src, src_gray, dst, dstMorph;
	
	src = imread("../Image/coin.jpg", 0);    

	checkLoad(src);

	printImg("source image", src);

	thresholdImg(src, dst, thresholdVal);

	morphImg(dst, dstMorph, "dilate", 13);

	printImg("dst image", dst);
	printImg("Morphology", dstMorph);

	waitKey(0);//Pause the program

	return 0;
}
