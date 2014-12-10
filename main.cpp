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

vector<int> calcuHaarFeature_sample(Mat sample, int *raw_feature, int compactSize) //提取单个样本特征的函数
{
    //=========================calculate the integral image================
    Mat inte;
    cv::integral(sample, inte);

    u32 * ptr_inte = (u32 *)inte.data;
    int status = calcuHaarFeature3(ptr_inte, sample.cols, sample.rows);
    vector<int> features(raw_feature, raw_feature+compactSize);
    return features;
}

vector<vector<int> > calcuHaarFeature_image(Mat image, int *raw_feature, int compactSize, int sample_cols, int sample_rows, int offset_x, int offset_y) //提取待检测图像特征的函数
{
    Mat inte;
    cv::integral(image, inte);

    vector<vector<int> > tile_feature;

    for(int i=0; i+sample_rows<image.rows; i+=offset_y)
        for(int j=0; j+sample_cols<image.cols; j+=offset_x)
        {
            Mat roi(image, Rect(j, i, sample_cols, sample_rows));
            Mat tile = roi.clone();

            u32 * ptr_inte = (u32 *)inte.data;
            int status = calcuHaarFeature3(ptr_inte, sample_cols, sample_rows);
            vector<int> features(raw_feature, raw_feature+compactSize);
            tile_feature.push_back(features);
        }

    return tile_feature;
}

//计算样本特征时如下调用
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
    prepare(&raw_feature, &compactSize, width, height); //分配内存，返回的是特征的存贮空间指针，以及特征的长度
                                                        //raw_feature 特征的存储空间的指针
                                                        //compactSize 特征的长度
                                                        //width, height 样本宽、高
                                                        
    cout<<"prepare complete"<<endl;

    vector<int> features = calcuHaarFeature_sample(sample, raw_feature, compactSize);
    //vector<int> features2 = calcuHaarFeature_sample(sample2, raw_feature, compactSize); //第二个样本同理

    //vector<vector<int> > all_sample_feature //将上面计算的features 全部存到这个容器中以便权哥的训练函数进行处理
    post_calculate(); //free memory

    return 0;
}

//如果是要计算带检测图像的特征如下调用
int main2(int argc, char *argv[])
{
    Mat image_grayscale = imread("image.jpg"); //待检测图像处理成灰度图

    int *raw_feature;
    int compactSize;
    int sample_cols = 21; // 我给出来的所有的样本都是这个尺寸，这也是检测器的尺寸
    int sample_rows = 28;
    prepare(&raw_feature, &compactSize, sample_cols, sample_rows); //分配内存，返回的是特征的存贮空间指针，以及特征的长度
                                                        //raw_feature 特征的存储空间的指针
                                                        //compactSize 特征的长度
                                                        //width, height 样本宽、高

    int offset_x = 6;
    int offset_y = 8; //检测器每次位移的像素个数

    vector<vector<int> > all_tile_features = calcuHaarFeature_image(image_grayscale, raw_feature, compactSize, sample_cols, sample_rows, offset_x, offset_y);
    //返回的是检测遍历图像得到的所有特征，把这些特征一次通过权哥代码训练出的检测器
    //需要将不同缩放大小的图像依次传给该函数，因为检测器只能检测特定大小的人脸

    post_calculate(); //free memory

    return 0;
}
