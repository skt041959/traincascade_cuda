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

vector<int> calcuHaarFeature_buffer(Mat sample, int compactSize)
{
    //=========================calculate the integral image================
    Mat inte;
    cv::integral(sample, inte);
    Mat inte2(inte, Rect(1, 1, 20, 20));
    Mat inte3 = inte2.clone();

    //如果输入的是待检测图，在这里分割成小块，尺寸和样本一样
    //待添加

    u32 * ptr_inte = (u32 *)inte.data;
    //cout<<inte.rows<<" "<<inte.cols<<endl;
    int *p_featureValue;
    //======================the entry of the feature calculating==============
    //这个函数用来计算样本特征和待检测图特征，需要对所有样本和带检测图应用此函数
    int status = calcuHaarFeature3(ptr_inte, &p_featureValue, sample.cols, sample.rows);
    vector<int> features(p_featureValue, p_featureValue+compactSize); //构建vector来跟权哥的程序对接
    return features;
}

int main(int argc, char *argv[])
{
    //==========================prepare sample image========================
    //Mat sample = imread("sample.bmp");
    Mat sample = cv::Mat::zeros(20, 20, CV_8UC1);
    for(int i=0; i<20; ++i)
    {
        sample.col(i).setTo(2*i);
    }
    //for(int i=0; i<sample.rows; ++i)
    //{
    //    for(int j=0; j<sample.cols; ++j)
    //    {
    //        cout<<(int)*(ptr_sample+i*sample.cols+j)<<"\t";
    //    }
    //    cout<<endl;
    //}
    //cout<<endl;
    //
    cout<<sample.rows<<" "<<sample.cols<<endl;
    int width = sample.cols;
    int height = sample.rows;
    u8 * ptr_sample = (u8 *)sample.data;

    int *raw_feature;
    int compactSize;
    prepare(&raw_feature, &compactSize, width, height);

    vector<int> features = calcuHaarFeature_buffer(sample, compactSize);

    post_calculate(); //free memory

    return 0;
}
