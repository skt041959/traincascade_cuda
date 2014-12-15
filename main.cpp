#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include "adaboost.hpp"
#include "image.hpp"

#include<fstream>
using namespace std;

int save_train(int pos,int nag,int clsifynum,int *clas);
float use_train(float *testx,int *testy,int *clasy,int samplenum,int *clas);

int read_sample(float *sample, int num, bool pos, bool test)
{
    char filename[50];
    for(int i=0; i<num; ++i)
    {
        if(!test)
            if(pos)
            {
                if(i<472)
                    sprintf(filename, "./bin/test_pos_feature_bin_%04d", i);
                else
                    sprintf(filename, "./bin/pos_feature_bin_%04d", i);
            }
            else
                sprintf(filename, "./bin/nag_feature_bin_%04d", i+400);
        else
            if(pos)
                sprintf(filename, "./bin/pos_feature_bin_%04d", i);
            else
                sprintf(filename, "./bin/test_nag_feature_bin_%04d", i);

        FILE *fp = fopen(filename, "rb");
        if(fp)
            fread((void*)sample, sizeof(float), featurelen1, fp);
        else
        {
            cout<<"invalid filename"<<endl;
            exit(1);
        }
        sample += featurelen1;
        fclose(fp);
    }
    return 0;
}

#ifndef MAKE_SAMPLE
int deprecated_main()
{
    //特征维度最不能超过131072，否则需要调整testcu.cu中threadofblock2的参数。
    //int featurelen = featurelen1;//特征维度，在testcu.h设置
    int clsifynum = clsifynum1;//分类器数目，在testcu.h设置
    int samplenum,testnum;//训练样本数及测试样本数

    float *tx;
    int *ty;
    int pos = 2200;
    int nag = 1500;
    int clas[2]= {0,1};//两种类别0和1；
    float testerror;
    //训练
    int t;
    cout<<"start train"<<endl;
    //t = save_train(pos,nag,clsifynum,clas);
    cout<<"classifier num:"<<t<<endl;


    //检测
    float *testx;
    int *testy;
    int pos_test = 200;
    int nag_test = 200;
    testnum = pos_test+nag_test;
    testx = (float*)malloc((testnum)*featurelen1*sizeof(float));
    cout<<"read test pos"<<endl;
    read_sample(testx, pos_test, true, true);
    cout<<"read test nag"<<endl;
    read_sample(testx+pos_test*featurelen1, nag_test, false, true);
    testy = (int*)malloc((pos_test+nag_test)*sizeof(int));
    for(int n=0; n<pos_test; n++)
        *(testy+n) = 1;

    for(int n=0; n<nag_test; n++)
        *(testy+n+pos_test) = 0;
    int *clasy = (int *)malloc(testnum*sizeof(int));
    cout<<"start test"<<endl;

    testerror = use_train(testx,testy,clasy,testnum,clas);

    for(int i=0; i<testnum; i++)
        cout<<"clasy "<<*(clasy+i)<<endl;
    cout<<"the final test error:"<<testerror<<endl;
    return 0;
}
#endif

int main(int argc, char *argv[])
{
    if(argc < 1)
        exit(1);

    float *feature;
    vector<Tile> place;
    int feature_rows = prepare_image(argv[1], &feature, place);

    int * flags = (int *)malloc(feature_rows*sizeof(int));
    for(int i=0; i<feature_rows; i++)
        *(flags+i) = 0;
    int * clasy = (int *)malloc(feature_rows*sizeof(int));
    for(int i=0; i<feature_rows; i++)
        *(clasy+i) = 0;

    int clas[2]= {0,1};

    float testerror = use_train(feature, flags, clasy, feature_rows, clas);

    vector<Tile> faces;
    for(int i=0; i<feature_rows; i++)
        if(*(clasy+i))
            faces.push_back(place[i]);

    show_image(argv[1], faces);

    return 0;

}

int save_train(int pos,int nag,int clsifynum,int *clas)
{
    int t;
    float alpha[clsifynum];//the weight of classifiers
    float thresh[clsifynum];//各个分类器的阈值
    vector< vector<int> > bia_fea(clsifynum,vector<int>(2));//偏置与特征列
    float *tx;
    int *ty;
    int samplenum = pos+nag;
    tx = (float*)malloc((pos+nag)*featurelen1*sizeof(float));
    cout<<"read pos"<<endl;
    read_sample(tx, pos, true, false);
    cout<<"read nag"<<endl;
    read_sample(tx+pos*featurelen1, nag, false, false);
    ty = (int*)malloc((pos+nag)*sizeof(int));
    for(int n=0; n<pos; n++)
        *(ty+n) = 1;

    for(int n=0; n<nag; n++)
        *(ty+n+pos) = 0;

    int featurelen = featurelen1;
    t = adatrain(tx,ty,alpha,thresh,bia_fea,clsifynum,featurelen,clas,samplenum);
    ofstream myfile1("thresh.txt");
    ofstream myfile2("bia_fea.txt");
    ofstream myfile3("alpha.txt");

    myfile1<<t<<endl;//保存t于thresh.txt中.
    for(int i=0;i<t;++i)
    {
        myfile1<<thresh[i]<<endl;
        myfile2<<bia_fea[i][0]<<" "<<bia_fea[i][1]<<endl;
        myfile3<<alpha[i]<<endl;
    }
    myfile1.close();
    myfile2.close();
    myfile3.close();
    free(tx);
    free(ty);
    return t;
}

float use_train(float *testx,int *testy,int *clasy,int samplenum,int *clas)
{
    ifstream myfile1("thresh.txt");
    ifstream myfile2("bia_fea.txt");
    ifstream myfile3("alpha.txt");
    float testerror;
    int t;
    myfile1>>t;

    float alpha[t];//the weight of classifiers
    float thresh[t];//各个分类器的阈值
    vector< vector<int> > bia_fea(t,vector<int>(2));//偏置与特征列


    for(int i=0;i<t;++i)
    {
        myfile1>>thresh[i];
        myfile2>>bia_fea[i][0];
        myfile2>>bia_fea[i][1];
        myfile3>>alpha[i];
    }
    myfile1.close();
    myfile2.close();
    myfile3.close();

    adaboostclassfy(testx,clasy,alpha,thresh,bia_fea,samplenum,t,clas);
    testerror = calerrorrate(testy,clasy,samplenum);
    return testerror;
}

