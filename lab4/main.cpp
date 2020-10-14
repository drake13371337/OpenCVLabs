#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int top = 300;
int base = 900;
int hight = 150;
int bot = 290;

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

int video_loop(string video_name, int frame_count, int frame_fps, int frame_width, int frame_height){
    Mat frame;

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

        vector<Point2f> out;
        vector<Point2f> res;

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

        imshow("Fragment", conv);
        trapeze(frame, res);

        msec+=1000/frame_fps;
        ++frame_num;

        string inf1 = "FPS: "+to_string(frame_fps)+"  Size: "+to_string(frame_width)+"x"+to_string(frame_height);
        string inf2 = "Frame: "+to_string(frame_num)+"/"+to_string(frame_count)+"  msec:"+to_string(msec);

        putText(frame, inf1, Point(50, 50), FONT_HERSHEY_SIMPLEX, frame_width/frame_height, Scalar(0, 255, 255), 2);
        putText(frame, inf2, Point(50, 100), FONT_HERSHEY_SIMPLEX, frame_width/frame_height, Scalar(0, 255, 255), 2);

        imshow("Video", frame);

        char c = (char)waitKey(1000/frame_fps);
        if (c == 27) return 0;
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
    createTrackbar("Top", "Video", &top, frame_width);
    createTrackbar("Base", "Video", &base, frame_width);
    createTrackbar("Hight", "Video", &hight, frame_height);
    createTrackbar("Bot", "Video", &bot, frame_height/2);

    video_loop(video_name, frame_count, frame_fps, frame_width, frame_height);

    return 0;
}

int main()
{
    string video_name = "challenge.mp4";

    video_action(video_name);
    
    return 0;
}
