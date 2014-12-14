//int fade_main() ;
#include<vector>
#include<iostream>
using namespace std;
#define featurelen1 43488 //特征维度
#define clsifynum1 600//分类器数目

/*float adatrain(vector< vector<float> > &trainx,vector<int> &trainy,float *alpha,float *thresh,
				vector< vector<int> >&bia_fea,int clsifynum,int featurelen,int *clas,int samplenum);
*/

int adatrain(float *h_x,int *h_y,float *alpha,float *thresh,
				vector< vector<int> >&bia_fea,int clsifynum,int featurelen,int *clas,int samplenum);				
				
/*void adaboostclassfy(vector< vector<float> > &trainx,vector<int> &testy,float *alpha,
						float *thresh,vector< vector<int> >&bia_fea,int samplenum,int t,int *clas);
						*/
void adaboostclassfy(float *trainx,int *testy,
						float *alpha,float *thresh,vector< vector<int> >&bia_fea,int samplenum,int t,int *clas);
						
/*float calerrorrate(vector<int> &trainy,vector<int> &testy,int samplenum);*/
float calerrorrate(int *trainy,int *testy,int samplenum);
