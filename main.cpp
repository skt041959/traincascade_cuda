#include<iostream>
#include <stdlib.h>
#include<vector>
#include "adaboost.hpp"
#include "image.hpp"
using namespace std;

int main()
{	
	
	//特征维度最不能超过131072，否则需要调整testcu.cu中threadofblock2的参数。
	int featurelen = featurelen1;//特征维度，在testcu.h设置
	int clsifynum = clsifynum1;//分类器数目，在testcu.h设置
	
    int samplenum,testnum;//训练样本数及测试样本数
    vector< vector<float> >  tx;//训练样本
    vector<int> ty;//训练样本实际种类

    prepare_sample(tx, ty);
    //cout<<tx.size()<<endl;
    //cout<<tx[0].size()<<endl;
    samplenum = tx.size();
    featurelen = tx[0].size();

    vector< vector<float> >  testx(testnum);//测试样本
    vector<int> testy(testnum);//测试样本实际类别
	//以上为需要初始值的部分

	//以下不需设置
	int clas[2]= {0,1};//两种类别0和1；
	float alpha[clsifynum];//the weight of classifiers
    float thresh[clsifynum];//各个分类器的阈值
    vector< vector<int> > bia_fea(clsifynum,vector<int>(2));//偏置与特征列
	float testerror;
	int t;
	//训练
    t = adatrain(tx,ty,alpha,thresh,bia_fea,clsifynum,featurelen,clas,samplenum);
	cout<<"classifier num:"<<t<<endl;
	//检测
    vector<int> clasy(testnum);
    adaboostclassfy(testx,clasy,alpha,thresh,bia_fea,testnum,t,clas);
    testerror = calerrorrate(testy,clasy,testnum);
    cout<<"the final test error:"<<testerror<<endl;
    return 0;

}


