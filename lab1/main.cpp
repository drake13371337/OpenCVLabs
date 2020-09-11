#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

int main()
{
    Mat input_image = imread("image.jpg", IMREAD_COLOR);
    Mat HSV_image, Grayscale_image, bin_image, pnt_image;
    Mat image = input_image.clone();

    cvtColor(input_image, HSV_image, COLOR_BGR2HSV);
    cvtColor(input_image, Grayscale_image, COLOR_BGR2GRAY);
    threshold(input_image, bin_image, 190, 255, THRESH_BINARY);

    rectangle(image, Point(113,114), Point(144,6), Scalar(255,0,0), 2);
    rectangle(image, Point(76,107), Point(102,81), Scalar(255,0,0), 2);
    rectangle(image, Point(356,26), Point(390,80), Scalar(255,0,0), 2);
    rectangle(image, Point(188,108), Point(206,91), Scalar(255,0,0), 2);
    rectangle(image, Point(36,273), Point(84,207), Scalar(0,255,0), 2);

    imshow("Input image", input_image);
    imshow("HSV image", HSV_image);
    imshow("Gray image", Grayscale_image);
    imshow("Bin image", bin_image);
    imshow("Image", image);

    waitKey();
    return 0;
}
