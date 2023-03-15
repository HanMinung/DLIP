/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Histogram Demo (from OpenCV docs)
* https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
* Created: 2023-Spring
------------------------------------------------------*/

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // Load image
    // CommandLineParser parser(argc, argv, "{@input | ../../Image/coin.jpg | input image}");
    // Mat src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    Mat src = imread("../../Image/coin.jpg", 0);    // Load an image
    
    if (src.empty())
    {
        return EXIT_FAILURE;
    }

    // Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    // Establish the number of bins
    int histSize = 256;

    // Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 };         // the upper boundary is exclusive
    const float* histRange = { range };


    // Set histogram param
    bool uniform = true, accumulate = false;

    // Compute the histograms
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate); // B-scale
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate); // G-scale
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate); // R-scale

    // Draw the histograms for B, G and R
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // Normalize the result to ( 0, histImage.rows )
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Draw for each channel
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1),  hist_h - cvRound(b_hist.at<float>(i - 1))),
                        Point(bin_w * (i),      hist_h - cvRound(b_hist.at<float>(i))),
                        Scalar(255, 0, 0), 2, 8, 0);

        line(histImage, Point(bin_w * (i - 1),  hist_h - cvRound(g_hist.at<float>(i - 1))), // img, pt1
                        Point(bin_w * (i),      hist_h - cvRound(g_hist.at<float>(i))),     // pt2
                        Scalar(0, 255, 0), 2, 8, 0);                                        // color, thickenss, linetype, shift

        line(histImage, Point(bin_w * (i - 1),  hist_h - cvRound(r_hist.at<float>(i - 1))),
                        Point(bin_w * (i),      hist_h - cvRound(r_hist.at<float>(i))),
                        Scalar(0, 0, 255), 2, 8, 0);
    }

    // Display
    imshow("Source image", src);
    imshow("calcHist Demo", histImage);
    waitKey();

    return EXIT_SUCCESS;
}
