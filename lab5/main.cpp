#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <set>

#include <iostream>

using namespace cv;
using namespace std;

int top = 350;
int base = 950;
int hight = 150;
int bot = 290;

int first = 220;
int second = 255;

const int R_WIDTH = 640;
const int R_HEIGHT = 480;

Mat conv(R_HEIGHT, R_WIDTH, CV_8UC3);

vector<Point2f> trapeze(Mat& inp, vector<Point2f> res){
    line(inp, res[0], res[1], Scalar(0, 255, 255), 2);
    line(inp, res[1], res[3], Scalar(0, 255, 255), 2);
    line(inp, res[3], res[2], Scalar(0, 255, 255), 2);
    line(inp, res[2], res[0], Scalar(0, 255, 255), 2);

    return res;
}

vector<Point> custom_findPoints(Mat input, int k){
    vector<Point> res;
    for(int i=0;i<input.rows;i++){
        for(int j=0;j<input.cols;j++){
            if(!(input.at<uchar>(j, i)<k))
                res.push_back(Point(j, i));
        }
    }
    return res;
}

vector<Point> eps_points(vector<Point> points, int eps){
    vector<Point> next_points, sum_points;
    vector<Point> cur_points(points);
    long int xbuff, ybuff;
    int indexp = 0;

    while(true){
        if(indexp>=(int)cur_points.size())
            return cur_points;

        for(int i=0;i<(int)cur_points.size();i++){
            if(i!=indexp){
                if( (sqrt( pow( cur_points.at(indexp).x-cur_points.at(i).x, 2) + pow( cur_points.at(indexp).y-cur_points.at(i).y, 2)))<eps ){
                    sum_points.push_back(cur_points.at(i));
                } else {
                    next_points.push_back(cur_points.at(i));
                }
            } else {
                sum_points.push_back(cur_points.at(i));
            }
        }

        if(sum_points.size()==1){
            if(indexp==(int)(cur_points.size()-1)){
                next_points.insert(next_points.begin() ,cur_points.at(indexp));
                return next_points;
            } else {
                indexp++;
            }
        }

        if(sum_points.size()>1){
            xbuff=sum_points.at(0).x;
            ybuff=sum_points.at(0).y;
            for(int k=1;k<(int)sum_points.size();k++){
                xbuff=xbuff+sum_points.at(k).x;
                ybuff=ybuff+sum_points.at(k).y;
            }
            xbuff=xbuff/sum_points.size();
            ybuff=ybuff/sum_points.size();
            next_points.insert(next_points.begin() ,Point(xbuff, ybuff));
            indexp++;
        }

        cur_points.clear();
        for(Point a : next_points)
            cur_points.push_back(a);
        next_points.clear();
        sum_points.clear();
    }
    return points;
}

vector<Point> fix_points(vector<Point> points, int eps){
    vector<Point> cur_points(points);
    vector<Point> res, next_points;
    int a = 0;
    int b = 0;

    for(int j=0;j<(int)cur_points.size();j++){
        for(int i=j+1;i<(int)cur_points.size();i++){
            if( (sqrt( pow( cur_points.at(j).x-cur_points.at(i).x, 2) + pow( cur_points.at(j).y-cur_points.at(i).y, 2)))<eps ){
                for(int k=0;k<(int)next_points.size();k++){
                    if(next_points.at(k)==cur_points.at(i))
                        a++;
                    if(next_points.at(k)==cur_points.at(j))
                        b++;
                }
                if(a==0)
                    next_points.push_back(cur_points.at(i));
                if(b==0)
                    next_points.push_back(cur_points.at(j));
                a=0;
                b=0;
            }
        }
    }
    for(Point a : next_points)
        res.push_back(a);
    return res;
}

vector<Point> line_search(Mat input, int sRows, int sCols, int eps, int eps1, int color){
    vector<Point> points, rect_points;
    Point buff;
    long int xbuff, ybuff;
    int xb = input.rows/sRows;
    int yb = input.cols/sCols;

    Mat sRect(sRows, sCols, CV_8U);

    for(int i=0;i<xb;i++){
        for(int j=0;j<yb;j++){
            for(int i1=0;i1<sRows;i1++){
                for(int j1=0;j1<sCols;j1++){
                    sRect.at<uchar>(i1, j1)=input.at<uchar>(i1+i*sRows, j1+j*sCols);
                }
            }

            rect_points = custom_findPoints(sRect, color);

            for(int r=0;r<(int)rect_points.size();r++){
                rect_points.at(r).x=rect_points.at(r).x+i*sRows;
                rect_points.at(r).y=rect_points.at(r).y+j*sCols;
            }

            if(rect_points.size()>1){
                xbuff=rect_points.at(0).x;
                ybuff=rect_points.at(0).y;
                for(int k=1;k<(int)rect_points.size();k++){
                    xbuff=xbuff+rect_points.at(k).x;
                    ybuff=ybuff+rect_points.at(k).y;
                }
                xbuff=xbuff/rect_points.size();
                ybuff=ybuff/rect_points.size();
                points.push_back(Point(ybuff, xbuff));
            }

            if(rect_points.size()==1)
                points.push_back(rect_points.at(0));
            rect_points.clear();
        }
    }
    return eps_points(eps_points(fix_points(points, eps1), eps), eps);
}

int video_loop(string video_name, int frame_count, int frame_fps, int frame_width, int frame_height){
    Mat frame, bin_image, gray_image, blur_image;

    VideoCapture cap(video_name);
    if(!cap.isOpened()){
        cout<<"Could not open "<<video_name<<endl;
        return -1;
    }

    int frame_num = 0;
    int msec = 0;

    for(;;){
        cap>>frame;

        if(frame.empty()){
            cout<<"End"<<endl;
            break;
        }

        vector<Point2f> out, res;
        vector<Point> points;

        res.push_back(Point2f((int)(frame_width/2-top/2), (int)(frame_height/2-hight+bot)));
        res.push_back(Point2f((int)(frame_width/2+top/2), (int)(frame_height/2-hight+bot)));
        res.push_back(Point2f((int)(frame_width/2-base/2), (int)(frame_height/2+bot)));
        res.push_back(Point2f((int)(frame_width/2+base/2), (int)(frame_height/2+bot)));

        out.push_back(Point2f(0, 0));
        out.push_back(Point2f(R_WIDTH, 0));
        out.push_back(Point2f(0, R_HEIGHT));
        out.push_back(Point2f(R_WIDTH, R_HEIGHT));

        Mat M = getPerspectiveTransform(res, out);
        warpPerspective(frame, conv, M, conv.size(), INTER_LINEAR, BORDER_CONSTANT);


        threshold(conv, bin_image, first, second, THRESH_BINARY);
        cvtColor(bin_image, gray_image, COLOR_BGR2GRAY);
        GaussianBlur(gray_image, blur_image, Size(11, 11), 1, 1);


        points = line_search(blur_image, 20, 20, 20, 40, 50);

        for(Point i : points){
            circle(conv, i, 5, Scalar(0, 0, 0), 3);
        }

        trapeze(frame, res);

        msec+=1000/frame_fps;
        ++frame_num;

        string inf1 = "FPS: "+to_string(frame_fps)+"  Size: "+to_string(frame_width)+"x"+to_string(frame_height);
        string inf2 = "Frame: "+to_string(frame_num)+"/"+to_string(frame_count)+"  msec:"+to_string(msec);

        putText(frame, inf1, Point(50, 50), FONT_HERSHEY_SIMPLEX, frame_width/frame_height, Scalar(0, 255, 255), 2);
        putText(frame, inf2, Point(50, 100), FONT_HERSHEY_SIMPLEX, frame_width/frame_height, Scalar(0, 255, 255), 2);

        imshow("Fragment", conv);
        imshow("Video", frame);
        imshow("Bin Fragment", blur_image);

        if (waitKey(27) >= 0)
            return 0;
    }
    cap.release();
    video_loop(video_name, frame_count, frame_fps, frame_width, frame_height);
    return 0;
}

int video_action(string video_name){
    VideoCapture cap(video_name);
    if(!cap.isOpened()){
        cout<<"Could not open "<<video_name<<endl;
        return -1;
    }

    int frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    int frame_fps = cap.get(CAP_PROP_FPS);
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);

    cap.release();

    namedWindow("Video", WINDOW_AUTOSIZE);
    namedWindow("Fragment", WINDOW_AUTOSIZE);
    namedWindow("Bin Fragment", WINDOW_AUTOSIZE);
    createTrackbar("Top", "Video", &top, frame_width);
    createTrackbar("Base", "Video", &base, frame_width);
    createTrackbar("Hight", "Video", &hight, frame_height);
    createTrackbar("Bot", "Video", &bot, frame_height/2);
    createTrackbar("1", "Video", &first, 255);
    createTrackbar("2", "Video", &second, 255);

    video_loop(video_name, frame_count, frame_fps, frame_width, frame_height);

    return 0;
}

int main()
{
    string video_name = "challenge.mp4";

    video_action(video_name);

    return 0;
}

