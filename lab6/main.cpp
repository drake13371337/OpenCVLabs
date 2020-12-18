#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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

vector<Point2f> custom_findPoints(Mat input, int k){
    vector<Point2f> res;
    for(int i=0;i<input.rows;i++){
        for(int j=0;j<input.cols;j++){
            if(!(input.at<uchar>(j, i)<k))
                res.push_back(Point2f(j, i));
        }
    }
    return res;
}

vector<Point2f> eps_points(vector<Point2f> points, int eps){
    vector<Point2f> next_points, sum_points;
    vector<Point2f> cur_points(points);
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
            next_points.insert(next_points.begin() ,Point2f(xbuff, ybuff));
            indexp++;
        }

        cur_points.clear();
        for(Point2f a : next_points)
            cur_points.push_back(a);
        next_points.clear();
        sum_points.clear();
    }
    return points;
}

vector<Point2f> fix_points(vector<Point2f> points, int eps){
    vector<Point2f> cur_points(points);
    vector<Point2f> res, next_points;
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
    for(Point2f a : next_points){
        res.push_back(a);
    }
    return res;
}

vector<Point2f> l_points(vector<Point2f> input){
    vector<Point2f> left;
    for(Point2f a : input)
        if(a.x<R_WIDTH/2)
            left.push_back(a);
    return left;
}

vector<Point2f> r_points(vector<Point2f> input){
    vector<Point2f> right;
    for(Point2f a : input)
        if(a.x>R_WIDTH/2)
            right.push_back(a);
    return right;
}

bool cmp(Point2f first, Point2f second) {
    return first.x < second.x;
}

vector<Point2f> approx(vector<Point2f> points, vector<float> ptr){
    vector<Point2f> res;
    const int size = points.size();
    float x[size];
    float y[size];
    for(int i=0;i<size;++i){
        x[i]=points[i].y;
        y[i]=points[i].x;
    }
    float sum, tmp, tmp_val;
    for(int i=0;i<size;i++)
        for (int j=i;j>=1;j--)
          if (x[j]<x[j-1]){
              tmp=x[j-1];
              x[j-1]=x[j];
              x[j]=tmp;
              tmp=y[j-1];
              y[j-1]=y[j];
              y[j]=tmp;
          }

    int cnt = 2;
    float arrs[size][size], fr[size];
    for(int i=0;i<cnt+1;i++)
        for(int j=0;j<cnt+1; j++){
            arrs[i][j]=0;
            for (int k=0; k<size; k++)
                arrs[i][j]+=pow(x[k], i+j);
        }

    for(int i=0;i<cnt+1;i++){
        fr[i]=0;
        for(int k=0;k<size;k++)
            fr[i]+=pow(x[k],i)*y[k];
    }
    for(int k=0;k<cnt+1;k++)
        for(int i=k+1;i<cnt+1;i++){
            tmp_val=arrs[i][k]/arrs[k][k];
            for(int j=k;j<cnt+1;j++)
                arrs[i][j]-=tmp_val*arrs[k][j];
            fr[i]-=tmp_val*fr[k];
        }

    vector<float> coeff(size);
    for(int i=cnt;i>=0;i--){
        sum = 0;
        for(int j=i;j<cnt+1;j++)
            sum+=arrs[i][j]*coeff[j];
        coeff[i]=(fr[i]-sum)/arrs[i][i];
    }

    for(auto y=ptr.at(0); y<ptr.back(); y+=0.5){
        float x = 0;
        for(int r=0;r<(int)coeff.size();r++)
            x+=coeff.at(r)*pow(y, r);
        res.push_back(Point2f(x, y));
    }
    return res;
}

vector<float> gen_points(int a, int b, int space){
    vector<float> res;
    int buff = 0;
    while(true){
        res.push_back(a+buff);
        buff+=space;
        if((a+buff)>=b)
            break;
    }
    res.push_back(b);
    return res;
}

vector<Point2f> line_search(Mat input, int sRows, int sCols, int eps, int eps1, int color){
    vector<Point2f> points, rect_points;
    Point2f buff;
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

vector<Point2f> trapeze_to_frame(vector<Point2f> points, int frame_width, int frame_height){
    vector<Point2f> out1, res1, rtpoints;

    out1.push_back(Point2f((int)(frame_width/2-top/2), (int)(frame_height/2-hight+bot)));
    out1.push_back(Point2f((int)(frame_width/2+top/2), (int)(frame_height/2-hight+bot)));
    out1.push_back(Point2f((int)(frame_width/2-base/2), (int)(frame_height/2+bot)));
    out1.push_back(Point2f((int)(frame_width/2+base/2), (int)(frame_height/2+bot)));

    res1.push_back(Point2f(0, 0));
    res1.push_back(Point2f(R_WIDTH, 0));
    res1.push_back(Point2f(0, R_HEIGHT));
    res1.push_back(Point2f(R_WIDTH, R_HEIGHT));

    Mat G = getPerspectiveTransform(res1, out1);
    for(Point a : points){
        rtpoints.push_back(Point2f( (a.x*G.at<double>(0, 0)+a.y*G.at<double>(0, 1)+G.at<double>(0, 2)) /
                                    (a.x*G.at<double>(2, 0)+a.y*G.at<double>(2, 1)+G.at<double>(2, 2)),
                                    (a.x*G.at<double>(1, 0)+a.y*G.at<double>(1, 1)+G.at<double>(1, 2)) /
                                    (a.x*G.at<double>(2, 0)+a.y*G.at<double>(2, 1)+G.at<double>(2, 2)) ));
    }
    return rtpoints;
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

        vector<Point2f> out, res, out1, res1, points, rpoints, lpoints, rline, lline, lstrline, lstlline;
        vector<float> x;

        //Frame -> Trapeze (frame -> conv)
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

        //Image proc (conv)
        threshold(conv, bin_image, first, second, THRESH_BINARY);
        cvtColor(bin_image, gray_image, COLOR_BGR2GRAY);
        GaussianBlur(gray_image, blur_image, Size(11, 11), 1, 1);

        //Line search (conv)
        points = line_search(blur_image, 20, 20, 20, 30, 50);

        //Draw cirlces (conv)
        for(Point i : points){
            circle(conv, i, 5, Scalar(0, 0, 0), 3);
        }
        //Trapeze(line points) -> Frame(points) (conv -> frame)
        rpoints = trapeze_to_frame(r_points(points), frame_width, frame_height);
        lpoints = trapeze_to_frame(l_points(points), frame_width, frame_height);

        //Draw cirlces (frame)
        /*for(Point i : rpoints){
            circle(frame, i, 5, Scalar(0, 0, 0), 3);
        }
        for(Point i : lpoints){
            circle(frame, i, 5, Scalar(0, 0, 0), 3);
        }*/

        //Draw line (frame)
        x = gen_points(frame_height/2-hight+bot, frame_height/2+bot, 10);
        sort(rpoints.begin(), rpoints.end(), cmp);
        sort(lpoints.begin(), lpoints.end(), cmp);
        
        if(!rpoints.empty()){
            rline = approx(rpoints, x);
            lstrline = rline;
        } else { rline = lstrline; }
        if(!lpoints.empty()){
            lline = approx(lpoints, x);
            lstlline = lline;
        } else { lline = lstlline; }

        for(Point i : rline){
            circle(frame, i, 5, Scalar(0, 255, 0), 3);
        }
        for(Point i : lline){
            circle(frame, i, 5, Scalar(0, 255, 0), 3);
        }

        //Draw info (frame)
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

int main(){
    string video_name = "challenge.mp4";

    video_action(video_name);

    return 0;
}
