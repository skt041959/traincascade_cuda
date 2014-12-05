#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "haar.hpp"

using namespace cv;
using namespace std;

int calcuHaarFeature_buffer(u32 *ptr, vector<SFeature> & features, int width, int height)
{
    int s = calcuHaarFeature(ptr, features, width, height);
    return s;
}

int main(int argc, char *argv[])
{
    //Mat sample = imread("sample.bmp");
    Mat sample = cv::Mat::zeros(20, 20, CV_8UC1);
    for(int i=0; i<20; ++i)
    {
        sample.col(i).setTo(2*i);
    }
    cout<<sample.rows<<" "<<sample.cols<<endl;
    int width = sample.cols;
    int height = sample.rows;
    //Mat inte = Mat::zeros(sample.cols+1, sample.rows+1, CV_8UC1);
    Mat inte;

    //integral_cuda(ptr, ptr_inte, width, height);
    cv::integral(sample, inte);
    Mat inte2(inte, Rect(1, 1, 20, 20));
    Mat inte3 = inte2.clone();

    u32 * ptr_inte = (u32 *)inte.data;
    u8 * ptr_sample = (u8 *)sample.data;
    cout<<inte.rows<<" "<<inte.cols<<endl;
    //for(int i=0; i<inte.rows; ++i)
    //{
    //    for(int j=0; j<inte.cols; ++j)
    //    {
    //        cout<<*(ptr_inte+i*inte.cols+j)<<"\t";
    //    }
    //    cout<<endl;
    //}
    for(int i=0; i<sample.rows; ++i)
    {
        for(int j=0; j<sample.cols; ++j)
        {
            cout<<(int)*(ptr_sample+i*sample.cols+j)<<"\t";
        }
        cout<<endl;
    }
    cout<<endl;

    vector<SFeature>features;
    int status = calcuHaarFeature_buffer(ptr_inte, features, width, height);
    cout<<features.size()<<endl;
    //for(size_t i=0; i<features.size(); ++i)
    //{
    //    for(int j=0; j<features[i].featureNum; ++j)
    //        cout<<*(features[i].pfeature+j)<<" ";
    //    cout<<endl;
    //}
    //cout<<endl;

    return 0;
}
