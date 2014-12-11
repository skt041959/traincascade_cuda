#include <iostream>
#include <stdlib.h>
#include <cstdio>
#include <vector>

#include "adaboost.hpp"
#include "image.hpp"
using namespace std;

#define MAKE_SAMPLE


int read_sample(float *sample, int num, bool pos, bool test)
{
    char filename[50];
    for(int i=0; i<num; ++i)
    {
        if(!test)
            if(pos)
                sprintf(filename, "pos_feature_bin_%04d", i);
            else
                sprintf(filename, "nag_feature_bin_%04d", i);
        else
            if(pos)
                sprintf(filename, "test_pos_feature_bin_%04d", i);
            else
                sprintf(filename, "test_nag_feature_bin_%04d", i);

        FILE *fp = fopen(filename, "rb");
        fread((void*)sample, sizeof(float), num, fp);
        sample += featurelen1;
        fclose(fp);
    }
    return 0;
}

#ifndef MAKE_SAMPLE
int main()
{	
	
	//特征维度最不能超过131072，否则需要调整testcu.cu中threadofblock2的参数。
	//int featurelen = featurelen1;//特征维度，在testcu.h设置
	int clsifynum = 600;//分类器数目，在testcu.h设置
	
    int samplenum,testnum;//训练样本数及测试样本数

    float *tx;
    int *ty;

    int pos = 500;
    int nag = 1000;
    samplenum = pos+nag;
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

    //samplenum = prepare_sample(&tx, &ty);
    //cout<<tx.size()<<endl;
    //cout<<tx[0].size()<<endl;
    //samplenum = tx.size();
    //featurelen = tx[0].size();
    int featurelen = featurelen1;

	int clas[2]= {0,1};//两种类别0和1；
	float alpha[clsifynum];//the weight of classifiers
    float thresh[clsifynum];//各个分类器的阈值
    vector< vector<int> > bia_fea(clsifynum,vector<int>(2));//偏置与特征列
	float testerror;
	int t;
	//训练
    cout<<"start train"<<endl;
    t = adatrain(tx,ty,alpha,thresh,bia_fea,clsifynum,featurelen,clas,samplenum);
	cout<<"classifier num:"<<t<<endl;
	//检测

    float *testx;
    int *testy;
    int pos_test = 100;
    int nag_test = 100;

    testx = (float*)malloc((pos_test+nag_test)*featurelen1*sizeof(float));
    cout<<"read test pos";
    read_sample(testx, pos, true, true);
    cout<<"read test nag";
    read_sample(testx+pos*featurelen1, nag, false, true);
    testy = (int*)malloc((pos_test+nag_test)*sizeof(int));

    for(int n=0; n<pos_test; n++)
        *(testy+n) = 1;

    for(int n=0; n<nag_test; n++)
        *(testy+n+pos_test) = 0;

    testnum = pos_test+nag_test;
    int *clasy = (int *)malloc(testnum*sizeof(int));
    adaboostclassfy(testx,clasy,alpha,thresh,bia_fea,testnum,t,clas);
    testerror = calerrorrate(testy,clasy,testnum);
    cout<<"the final test error:"<<testerror<<endl;

    return 0;
}
#endif


