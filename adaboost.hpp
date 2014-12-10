#include<vector>
#include<iostream>
using namespace std;
#define featurelen1 208
#define clsifynum1 60//分类器数目

float adatrain(vector< vector<float> > &trainx,vector<int> &trainy,float *alpha,float *thresh,
				vector< vector<int> >&bia_fea,int clsifynum,int featurelen,int *clas,int samplenum);

void adaboostclassfy(vector< vector<float> > &trainx,vector<int> &testy,float *alpha,
						float *thresh,vector< vector<int> >&bia_fea,int samplenum,int t,int *clas);
						
float calerrorrate(vector<int> &trainy,vector<int> &testy,int samplenum);
