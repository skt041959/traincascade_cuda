#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "haar.hpp"
#include "image.hpp"

using namespace cv;
using namespace std;

#define DIMENSION
//#define MAKE_SAMPLE

//vector<float> calcuHaarFeature_sample(Mat sample, float *raw_feature, int compactSize) //提取单个样本特征的函数
int calcuHaarFeature_sample(Mat sample, float *raw_feature, int compactSize) //提取单个样本特征的函数
{
    Mat equalized;
    cv::equalizeHist(sample, equalized);
    Mat inte;
    cv::integral(equalized, inte);

    u32 * ptr_inte = (u32 *)inte.data;
    int status = calcuHaarFeature3(ptr_inte, sample.cols, sample.rows);
    //vector<float> features(raw_feature, raw_feature+compactSize);
    //return features;
    return status;
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

vector<Mat> read_image_list(const char * filename, int num)
{
    char buffer[256];
    fstream list_file;
    list_file.open(filename, ios::in);
    vector<Mat> image_list;
    string pgm(".pgm");
    int n=0;
    while(!list_file.eof() && n<num)
    {
        list_file.getline(buffer, 256, '\n');
        string file(buffer);
        if(file.length()>4 && 0 == file.compare(file.length()-4, 4, pgm))
        {
            Mat t = imread(buffer, 0);
            if(t.cols != SAMPLE_COLS || t.rows != SAMPLE_ROWS)
            {
                printf("%s\n", buffer);
                cout<<t.cols<<","<<t.rows<<endl;
                exit(1);
            }
            image_list.push_back(t);
        }
        n++;
    }
    list_file.close();
    return image_list;
}

//int prepare_sample(vector<vector<float> > &sample, vector<int> &flag)
int prepare_sample(float **sample, int **flag, int pos_num, int nag_num, bool write)
{
    vector<Mat> pos_sample = read_image_list("./positive.txt", pos_num);
    vector<Mat> nag_sample = read_image_list("./nagetive.txt", nag_num);

    //vector<int> pos_flag(pos_sample.size(), 1);
    //vector<int> nag_flag(nag_sample.size(), 0);
    //flag.insert(flag.end(), pos_flag.begin(), pos_flag.end());
    //flag.insert(flag.end(), nag_flag.begin(), nag_flag.end());

    if(pos_sample.size() != pos_num || nag_sample.size() != nag_num)
    {
        cout<<"invalid sample num"<<endl;
        exit(1);
    }


    int num = pos_num+nag_num;
    int *f = (int*)malloc(num*sizeof(int));
    int *i=f;
    for(int n=0; n<pos_num; ++n)
    {
        *i=1;
        i++;
    }
    for(int n=0; n<nag_num; ++n)
    {
        *i=0;
        i++;
    }
    *flag = f;

    float *raw_feature;
    int compactSize;
    prepare(&raw_feature, &compactSize, SAMPLE_COLS, SAMPLE_ROWS);

    float *s = (float*)malloc(num*compactSize*sizeof(float));
    *sample = s;

    cout<<"pos feature"<<endl;
    int t=2000;
    char filename[50];
    for(vector<Mat>::iterator i=pos_sample.begin(); i!=pos_sample.end(); ++i)
    {
        //vector<float> features = calcuHaarFeature_sample(*i, raw_feature, compactSize);
        calcuHaarFeature_sample(*i, raw_feature, compactSize);
        //sample.push_back(features);
        memcpy(s, raw_feature, compactSize*sizeof(float));
        s+=compactSize;
        if(write)
        {
            sprintf(filename, "./bin/pos_feature_bin_%04d", t);
            FILE *fp = fopen(filename, "wb");
            fwrite(raw_feature, sizeof(float), compactSize, fp);
            fclose(fp);
        }
        t++;
        if(t%10==0)
            cout<<"\rpos "<<t<<flush;
    }
    cout<<" "<<pos_sample.size()<<endl;

    cout<<"nag feature"<<endl;
    t=4000;
    for(vector<Mat>::iterator i=nag_sample.begin(); i!=nag_sample.end(); ++i)
    {
        //vector<float> features = calcuHaarFeature_sample(*i, raw_feature, compactSize);
        calcuHaarFeature_sample(*i, raw_feature, compactSize);
        //sample.push_back(features);
        memcpy(s, raw_feature, compactSize*sizeof(float));
        s+=compactSize;
        if(write)
        {
            sprintf(filename, "./bin/nag_feature_bin_%04d", t);
            FILE *fp = fopen(filename, "wb");
            fwrite(raw_feature, sizeof(float), compactSize, fp);
            fclose(fp);
        }
        t++;
        if(t%10==0)
            cout<<"\rnag "<<t<<flush;
    }
    cout<<" "<<nag_sample.size()<<endl;

    post_calculate();
    cout<<"feature complete"<<endl;

    return num;
}

int prepare_test(float **sample, int **flag, int pos_num, int nag_num, bool write)
{
    vector<Mat> pos_sample = read_image_list("./test_positive.txt", pos_num);
    vector<Mat> nag_sample = read_image_list("./test_nagetive.txt", nag_num);

    //vector<int> pos_flag(pos_sample.size(), 1);
    //vector<int> nag_flag(nag_sample.size(), 0);
    //flag.insert(flag.end(), pos_flag.begin(), pos_flag.end());
    //flag.insert(flag.end(), nag_flag.begin(), nag_flag.end());

    if(pos_sample.size() != pos_num || nag_sample.size() != nag_num)
    {
        cout<<pos_sample.size()<<" "<<nag_sample.size();
        cout<<"invalid sample num"<<endl;
        exit(1);
    }


    int num = pos_num+nag_num;
    int *f = (int*)malloc(num*sizeof(int));
    int *i=f;
    for(int n=0; n<pos_num; ++n)
    {
        *i=1;
        i++;
    }
    for(int n=0; n<nag_num; ++n)
    {
        *i=0;
        i++;
    }
    *flag = f;

    float *raw_feature;
    int compactSize;
    prepare(&raw_feature, &compactSize, SAMPLE_COLS, SAMPLE_ROWS);

    float *s = (float*)malloc(num*compactSize*sizeof(float));
    *sample = s;

    cout<<"pos feature"<<endl;
    int t=0;
    char filename[50];
    for(vector<Mat>::iterator i=pos_sample.begin(); i!=pos_sample.end(); ++i)
    {
        //vector<float> features = calcuHaarFeature_sample(*i, raw_feature, compactSize);
        calcuHaarFeature_sample(*i, raw_feature, compactSize);
        //sample.push_back(features);
        memcpy(s, raw_feature, compactSize*sizeof(float));
        s+=compactSize;
        if(write)
        {
            sprintf(filename, "./bin/test_pos_feature_bin_%04d", t);
            FILE *fp = fopen(filename, "wb");
            fwrite(raw_feature, sizeof(float), compactSize, fp);
            fclose(fp);
        }
        t++;
        if(t%10==0)
            cout<<"\rpos "<<t<<flush;
    }
    cout<<" "<<pos_sample.size()<<endl;

    cout<<"nag feature"<<endl;
    t=0;
    for(vector<Mat>::iterator i=nag_sample.begin(); i!=nag_sample.end(); ++i)
    {
        //vector<float> features = calcuHaarFeature_sample(*i, raw_feature, compactSize);
        calcuHaarFeature_sample(*i, raw_feature, compactSize);
        //sample.push_back(features);
        memcpy(s, raw_feature, compactSize*sizeof(float));
        s+=compactSize;
        if(write)
        {
            sprintf(filename, "./bin/test_nag_feature_bin_%04d", t);
            FILE *fp = fopen(filename, "wb");
            fwrite(raw_feature, sizeof(float), compactSize, fp);
            fclose(fp);
        }
        t++;
        if(t%10==0)
            cout<<"\rpos "<<t<<flush;
    }
    cout<<" "<<nag_sample.size()<<endl;

    post_calculate();
    cout<<"feature complete"<<endl;

    return num;
}

int prepare_image(char * filename, float **feature, vector<Tile> &place)
{
    Mat image = imread(filename, 0);
    //int i = image.rows/19 /1.25
    vector<Mat> tiles;
    int col = image.cols;
    int row = image.rows;
    cout<<"image"<<col<<"x"<<row<<endl;
    float s=1;

    while(col>=SAMPLE_COLS && row>=SAMPLE_ROWS)
    {
        for(int i=0; i<(row-SAMPLE_ROWS); i+=5)
            for(int j=0; j<(col-SAMPLE_COLS); j+=5)
            {
                cout<<i<<","<<j<<endl;
                Mat t(image, Rect(j, i, SAMPLE_COLS, SAMPLE_ROWS));
                place.push_back(Tile(j, i, SAMPLE_COLS/s, SAMPLE_COLS/s));
                Mat c = t.clone();
                tiles.push_back(c);
            }

        s *= 0.8;
        col = col*s;
        row = row*s;

        image.resize(row);
        cout<<"image"<<col<<"x"<<row<<endl;
    }
    cout<<"tile size"<<tiles.size()<<endl;

    float *raw_feature;
    int compactSize;
    prepare(&raw_feature, &compactSize, SAMPLE_COLS, SAMPLE_ROWS);

    float *f = (float*)malloc(tiles.size()*compactSize*sizeof(float));
    *feature = f;
    int n=0;
    for(vector<Mat>::iterator i=tiles.begin(); i!=tiles.end(); ++i)
    {
        calcuHaarFeature_sample(*i, raw_feature, compactSize);
        memcpy(f, raw_feature, compactSize*sizeof(float));
        f+=compactSize;
        n++;
        if(n%10==0)
            cout<<"\rcalcu "<<n<<flush;
    }

    return tiles.size();
}

int show_image(char *filename, vector<Tile> &faces)
{
    Mat image = imread(filename, 0);
    Mat image2 = image.clone();
    Mat image3 = image.clone();
    Mat prob = Mat::zeros(image.size(), CV_32SC1);
    Mat single = Mat::zeros(image.size(), CV_32SC1);
    Mat binary, binary2;

    for(size_t i=0; i<faces.size(); i++)
    {
        single.setTo(0);
        rectangle(prob, Point(faces[i].x, faces[i].y), Point(faces[i].x2, faces[i].y2),\
            Scalar(1), CV_FILLED);
        rectangle(image2, Point(faces[i].x, faces[i].y), Point(faces[i].x2, faces[i].y2),\
            Scalar(255), 1);
        if(faces[i].width == SAMPLE_ROWS && faces[i].height == SAMPLE_COLS)
            rectangle(image3, Point(faces[i].x, faces[i].y), Point(faces[i].x2, faces[i].y2),\
                    Scalar(255), 1);

        prob += single;
    }
    //normalize(prob, prob, 0, 255, NORM_MINMAX);
    //double max;
    //minMaxIdx(prob, NULL, &max);
    normalize(prob, binary, 0, 255, NORM_MINMAX, CV_8U);
    threshold(binary, binary2, 120, 255, CV_THRESH_BINARY);

    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Rect> rects;
    cv::findContours(binary2, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if( !contours.empty() && !hierarchy.empty() )
    {
        vector< vector<Point> >::iterator i = contours.begin();
        for( ; i!=contours.end(); i++)
        {
            Rect s = cv::boundingRect(*i);
            rects.push_back(s);
        }
    }

    for(size_t i=0; i<rects.size(); i++)
    {
        cout<<"face place"<<rects[i].x<<"x"<<rects[i].y<<" "<<rects[i].width<<"x"<<rects[i].height<<endl;
        rectangle(image, rects[i], Scalar(255), 1);
    }

    namedWindow("image", 0);
    namedWindow("image2", 0);
    namedWindow("image3", 0);
    namedWindow("binary", 0);
    imshow("image", image);
    imshow("image2", image2);
    imshow("image3", image3);
    imshow("binary", binary);
    //waitKey(0);
    while(1)
    {
        char ch = waitKey(0);
        if((ch & 0xFF) == 27)
            break;
    }
}

#ifdef MAKE_SAMPLE
int main()
{
    float *tx;
    int *ty;
    //prepare_sample(&tx, &ty, 429, 548, true);

    float *testx;
    int *testy;
    prepare_test(&testx, &testy, 472, 1, true);

    return 0;
}
#endif

