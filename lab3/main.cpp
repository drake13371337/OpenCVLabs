#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

using namespace cv;

int main()
{
    Mat input_frame;
    Mat gauss_frame;
    Mat green_frame;
    Mat hsv_frame;

    Mat blur_gray_frame;
    Mat xgrad_frame;
    Mat ygrad_frame;
    Mat canny_frame;

    VideoCapture cap;

    int deviceID = 0;
    int apiID = cv::CAP_ANY;

    cap.open(deviceID, apiID);

    if (!cap.isOpened()){
        std::cout<<"ERROR! Unable to open camera\n"<<std::endl;
        return -1;
    }
    for (;;){
        cap.read(input_frame);

        GaussianBlur(input_frame, gauss_frame, Size(9, 9), 1, 1);

        cvtColor(input_frame, hsv_frame, COLOR_BGR2HSV);
        inRange(hsv_frame, Scalar(60, 50, 50), Scalar(75, 255, 255), green_frame);

        cvtColor(gauss_frame, blur_gray_frame, COLOR_BGR2GRAY);
        Sobel(blur_gray_frame, xgrad_frame, CV_16SC1, 1, 0);
        Sobel(blur_gray_frame, ygrad_frame, CV_16SC1, 0, 1);
        Canny(xgrad_frame, ygrad_frame, canny_frame, 60, 120);

        imshow("Input", input_frame);
        imshow("Blur", gauss_frame);
        imshow("Green", green_frame);
        imshow("Canny", canny_frame);

        if (waitKey(5) >= 0)
            break;
    }
    return 0;
}
