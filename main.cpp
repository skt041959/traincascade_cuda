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

    vector<int> features = calcuHaarFeature_sample(sample, raw_feature, compactSize);

    post_calculate(); //free memory

    return 0;
}
