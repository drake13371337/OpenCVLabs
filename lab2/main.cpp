#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

void filter_image(Mat in_image, Mat& out_image, int x, int y, std::vector<int> kern, int mat, int gray_type){
    out_image = in_image.clone();

    Vec3b buff = {0, 0, 0};
    uchar gbuff = 0;

    int kern_rows = mat;
    int kern_cols = mat;

    int kern_sum = 0;
    for (auto& n : kern)
        kern_sum += abs(n);

    for(int i=x; i<out_image.rows; i++){
        for(int j=y; j<out_image.cols; j++){
            if(gray_type){
                for(int ik=0; ik<kern_rows; ik++){
                    for(int jk=0; jk<kern_cols; jk++){
                        gbuff = gbuff+in_image.at<uchar>(i-1+ik, j-1+jk) * kern[ik+jk*mat]/kern_sum;
                    }
                }
                out_image.at<uchar>(i, j) = gbuff;
                gbuff = 0;
            } else {
                for(int k=0; k<3; k++){
                    for(int ik=0; ik<kern_rows; ik++){
                        for(int jk=0; jk<kern_cols; jk++){
                            buff[k] = buff[k]+in_image.at<Vec3b>(i-1+ik, j-1+jk)[k] * kern[ik+jk*mat]/kern_sum;
                        }
                    }
                }
                out_image.at<Vec3b>(i, j) = buff;
                buff = {0, 0, 0};
            }
        }
    }
}

int main()
{
    Mat input_image = imread("image.jpg", IMREAD_COLOR);

    Mat gray_image;
    cvtColor(input_image, gray_image, COLOR_BGR2GRAY);

    Mat gauss_blur_image;
    Mat custom_filter_image;
    Mat grad_filter_image;

    GaussianBlur(input_image, gauss_blur_image, Size(5, 5), 1, 1);

    std::vector<int> kern = {0, 0, 1,
                             0, 1, 2,
                             1, 2, 3};
    filter_image(input_image, custom_filter_image, 0, 0, kern, 3, 0);

    kern = {1, 2, 1,
            0, 0, 0,
            -1, -2, -1};
    filter_image(gray_image, grad_filter_image, 0, 0, kern, 3, 1);

    imshow("Input image", input_image);
    imshow("GaussianBlur", gauss_blur_image);
    imshow("Custom filter", custom_filter_image);
    imshow("Grad filter", grad_filter_image);

    waitKey();
    return 0;
}
