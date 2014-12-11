#include<iostream>
#include <stdlib.h>
#include<vector>
#include "adaboost.hpp"
#include "image.hpp"
using namespace std;

int main()
{	
	
	//特征维度最不能超过131072，否则需要调整testcu.cu中threadofblock2的参数。
	//int featurelen = featurelen1;//特征维度，在testcu.h设置
	int clsifynum = 600;//分类器数目，在testcu.h设置
	
    int samplenum,testnum;//训练样本数及测试样本数

    float *tx;
    int *ty;

    samplenum = prepare_sample(&tx, &ty);
    //cout<<tx.size()<<endl;
    //cout<<tx[0].size()<<endl;
    //samplenum = tx.size();
    //featurelen = tx[0].size();
    int featurelen = 63960;

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

    testnum = prepare_test(&testx, &testy);

    int *clasy = (int *)malloc(testnum*sizeof(int));
    adaboostclassfy(testx,clasy,alpha,thresh,bia_fea,testnum,t,clas);
    testerror = calerrorrate(testy,clasy,testnum);
    cout<<"the final test error:"<<testerror<<endl;

    return 0;
}


