#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include "adaboost.hpp"
using namespace std;

__host__ __device__ inline float getx(float *x,int i,int j,int featurelen=featurelen1)
{
	return x[i*featurelen + j];
}

void initmat(float *m,int l,float value)
{
    for(int i=0; i<l; i++)
        m[i] = value;
}
void initmat2(int *m,int l,float value)
{
    for(int i=0; i<l; i++)
        m[i] = value;
}
void initmat3(vector<int> &testy,int l,int value)
{
    for(int i=0; i<l; i++)
        testy[i] = value;
}

float weaklearnerclassfy(float *trainx,int *trainy,int *testy,
							float thita,int bias,int samplenum,float *weight,int f,int *clas)
{
	float cnt=0;
	initmat2(testy,samplenum,clas[1]);
	for(int i=0;i<samplenum;i++)
	{
		if (bias*getx(trainx,i,f)  > bias*thita) // trainx[i][f]
		{
			testy[i]=clas[0];
		}
		if(testy[i] != trainy[i])
			cnt += weight[i];
	}
	return cnt;
}

__device__ float weaklearnerclassfy3(float *trainx,int *trainy,
							float thita,int bias,int samplenum,float *weight,int f,int *clas)
{
	float cnt=0;
	int tempclas;
	for(int i=0;i<samplenum;i++)
	{	
		tempclas = clas[1];
		if (bias*getx(trainx,i,f) > bias*thita)
		{
			tempclas = clas[0];
		}
		if(tempclas != trainy[i])
			cnt += weight[i];
	}
	return cnt;
}


__global__ void searchbestweaklearner2(float *trainx,int *trainy,
								float *weight,float *accuracy,float *thresh,int *biasvec,int samplenum,int *clas,int *d_whchfea,int featurelen)								
{	
	int dx=(blockIdx.x*blockDim.x)+threadIdx.x;
    int dy=(blockIdx.y*blockDim.y)+threadIdx.y;
    int tid= dy+dx*gridDim.y*blockDim.y;
	float temperror,maxf,minf,step,span,thita,nfmean = 0;
	float pfmean = 0;
	int cnt=0;
	int p = 1;
	if(tid < featurelen)
	{	
		accuracy[tid] = 1;
		for(int i=0;i<samplenum;i++)
		{	
			if(trainy[i] == clas[0])
			{
				pfmean += getx(trainx,i,tid);//trainx[i][tid];//选择第tid个特征，计算最佳弱分类器。
				cnt++;
			}
			else
				nfmean += getx(trainx,i,tid);//trainx[i][tid];
		}
		pfmean = pfmean/((float)cnt);
		nfmean = nfmean/((float)(samplenum - cnt));
		maxf = nfmean;
		minf = pfmean;
		if (pfmean > nfmean)
		{
			maxf = pfmean;
			minf = nfmean;
		}
		step = (maxf-minf)/7;
		for(int i=0;i<4;i++)//迭代次数
		{
			for(int j=0;j<8;j++)//每次迭代将搜索区域划分的片段
			{
				thita = minf + j*step;
				temperror = weaklearnerclassfy3(trainx,trainy,thita,1,samplenum,weight,tid,clas);//
				if(temperror>0.5)//错误率超过0.5则反向偏置
				{
					temperror = 1-temperror;
					p = -1;
				}
				else
				{	p = 1;}
				if(temperror<accuracy[tid])
				{
					accuracy[tid] = temperror;
					thresh[tid] = thita;
					biasvec[tid] = p;
					d_whchfea[tid] = tid;
				}
			}
			span = (maxf-minf)/8;
			maxf = thresh[tid] + span;
			minf = thresh[tid] - span;
			step = (maxf - minf)/7;
		}
	}
}

__global__ void redct_acury(float *d_clsy,float *d_interresult,int *location,int *real_location,int featurelen,int choice,
								float *d_tempthresh,int *d_tempbias,int *d_whchfea)
{
	unsigned int dx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y*blockDim.y + threadIdx.y;
	int tid= dy+dx*gridDim.y*blockDim.y;
	if(choice==1 && tid<featurelen)
	{
		location[tid] = tid;
		__syncthreads();
	}
	for (unsigned int s=blockDim.y/2; s>0; s>>=1)
    {
        if (dy < s)
        {
			if(((dx*blockDim.y+dy+s) < featurelen) && (d_clsy[dx*blockDim.y+dy] > d_clsy[dx*blockDim.y+dy+s]))
			{
				d_clsy[dx*blockDim.y+dy] = d_clsy[dx*blockDim.y+dy+s];
				location[dx*blockDim.y+dy] = location[dx*blockDim.y+dy+s];
			}
        }
        __syncthreads();
    }
    if (dy==0)
	{
		d_interresult[dx] = d_clsy[dx*blockDim.y];
		real_location[dx] = location[dx*blockDim.y];
		if(choice == 2)
		{
			d_tempthresh[0] = d_tempthresh[ real_location[dx] ];
			d_tempbias[0] = d_tempbias[ real_location[dx] ];
			d_whchfea[0] = d_whchfea[ real_location[dx] ];
		}
	}
}


void renewweight(float *weight,int samplenum,float alpha,int *trainy,int *testy)
{
    float factor=0;
    for(int i=0; i<samplenum; i++)
    {
		factor += weight[i]*exp(-alpha*((trainy[i]==testy[i])?1:-1));
    }
    for(int i=0; i<samplenum; i++)
    {
        weight[i] = (weight[i]/factor)*exp(-alpha*( (trainy[i]==testy[i])?1:-1 ) );
    }
}

void adaboostclassfy(float *trainx,int *testy,
						float *alpha,float *thresh,vector< vector<int> >&bia_fea,int samplenum,int t,int *clas)
{
    float thita,temph=0;
    int p,fcol;
	initmat2(testy,samplenum,clas[1]);
    for(int i=0; i<samplenum; i++)
    {
		temph=0;
        for(int j=0; j<t; j++)
        {
            thita = thresh[j];
            p = bia_fea[j][0];
            fcol = bia_fea[j][1];
            if(p*getx(trainx,i,fcol) > (p*thita))
            {
				temph +=  alpha[j];
            }
			else
			{
				temph -=  alpha[j];
			}
        }
        if(temph > 0)
            testy[i] = clas[0];
    }
}


float calerrorrate(int *trainy,int *testy,int samplenum)
{
	int cnt=0;
	for(int i=0;i<samplenum;i++)
	{
		if (trainy[i] != testy[i])
		{
		cnt++;
		}
	}
	return ((float)cnt) / ((float)samplenum);
}

void get_trsh_bia_fea(int *d_location,float *d_tempthresh,int *d_tempbias,int *d_whchfea,float *thresh,vector< vector<int> >&bia_fea,int t)
{
	int tempi[1];
	cudaMemcpy( thresh+t,d_tempthresh,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy( tempi,d_tempbias,sizeof(int),cudaMemcpyDeviceToHost);
	bia_fea[t][0] = tempi[0];
	cudaMemcpy( tempi,d_whchfea,sizeof(int),cudaMemcpyDeviceToHost);
	bia_fea[t][1] = tempi[0];
}


int adatrain(float *h_x,int *h_y,float *alpha,float *thresh,
				vector< vector<int> >&bia_fea,int clsifynum,int featurelen,int *clas,int samplenum)
{
	
	float weight[samplenum],errorrate,fnlerror;
    int t=0;
    int  *testy = (int *)malloc(samplenum*sizeof(int));
    initmat2(testy,samplenum,clas[1]);
	initmat(weight,samplenum,1/float(samplenum));//需要改为对device端的weight 初始化；
	//cuda 设备端参数设置
	float *d_x,*d_clsy,*d_weight,*d_tempthresh,*d_interresult,*d_result;
	int *d_y,*d_tempbias,*d_clas,*d_whchfea,*d_location,*location,*real_location;
	size_t sizex,sizey,sizec;
	int numBlocksx = 32;
    int numBlocksy = 16;
    int numThreadsPerBlockx = 16;
    int numThreadsPerBlocky = 16;
	sizex = samplenum*featurelen*sizeof(float);
	sizey = samplenum*sizeof(int);
	sizec = samplenum*sizeof(float);

	cudaMalloc((void **)&d_x , sizex);
	cudaMalloc((void **)&d_y , sizey);
	cudaMalloc((void **)&d_clsy , featurelen*sizeof(float));//每个特征对应分类器的识别率
	cudaMalloc((void **)&d_weight , sizec);
	cudaMalloc((void **)&d_tempthresh , featurelen*sizeof(float));
	cudaMalloc((void **)&d_tempbias , featurelen*sizeof(int));
	cudaMalloc((void **)&d_whchfea , featurelen*sizeof(int));
	cudaMalloc((void **)&d_clas , 2*sizeof(int));
	cudaMalloc((void **)&d_interresult , 256*sizeof(float));//中间值，其大小随dimGrid2调整
	cudaMalloc((void **)&location , featurelen*sizeof(int));
	cudaMalloc((void **)&real_location , 256*sizeof(int));//大小随dimGrid2调整
	cudaMalloc((void **)&d_location , sizeof(int));
	cudaMalloc((void **)&d_result , sizeof(float));
	
	/*
	h_x = (float *) malloc(sizex);
	h_y = (int *)malloc(sizey);
	
	for(int i=0;i<samplenum;i++)
	{
		for(int j=0;j<featurelen;j++)
		{
			h_x[i*featurelen+j] = trainx[i][j];
		}
		h_y[i] = trainy[i];	
	}
	*/

	cudaMemcpy(d_x,h_x,sizex,cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,h_y,sizey,cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight,weight,sizec,cudaMemcpyHostToDevice);
	cudaMemcpy(d_clas,clas,2*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 dimGrid( numBlocksx,numBlocksy);
    dim3 dimBlock( numThreadsPerBlockx,numThreadsPerBlocky);
	int threadofblock2 = 512;//实际应用只需调整此处!
	dim3 dimGrid2( 256,1);
    dim3 dimBlock2( 1,threadofblock2);
	
	dim3 dimGrid3( 1,1);
    dim3 dimBlock3( 1,256);//严格匹配dimGrid2中的block数；
	int numblock3 = (featurelen+threadofblock2-1)/threadofblock2 ;
    while(1)
    {	
		cudaMemcpy(d_weight,weight,sizec,cudaMemcpyHostToDevice);

		searchbestweaklearner2<<<  dimGrid, dimBlock >>>(d_x,d_y,d_weight,d_clsy,d_tempthresh,d_tempbias,samplenum,d_clas,d_whchfea,featurelen);
		redct_acury<<<  dimGrid2, dimBlock2 >>>(d_clsy,d_interresult,location,real_location,featurelen,1,d_tempthresh,d_tempbias,d_whchfea);
		redct_acury<<<  dimGrid3, dimBlock3 >>>(d_interresult,d_result,real_location,d_location,numblock3,2,d_tempthresh,d_tempbias,d_whchfea);
		get_trsh_bia_fea(d_location,d_tempthresh,d_tempbias,d_whchfea,thresh,bia_fea,t);
		
        errorrate = weaklearnerclassfy(h_x,h_y,testy,thresh[t],bia_fea[t][0],samplenum,weight,bia_fea[t][1],clas);//使用t轮最优分类器进行分类trainx未修改
		cout<<"权重错误:"<<errorrate<<endl;
        alpha[t] = 0.5 * log((1-errorrate)/errorrate);
        renewweight(weight,samplenum,alpha[t],h_y,testy);
        adaboostclassfy(h_x,testy,alpha,thresh,bia_fea,samplenum,t+1,clas);//changed
		fnlerror = calerrorrate(h_y,testy,samplenum);
		cout<<"t:"<<t<<"  train error:"<<fnlerror<<endl;
		t++;
        if(t >= clsifynum || fnlerror < 0.0001)
        {
			break;
		}
    }
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_clsy);
	cudaFree(d_weight);
	cudaFree(d_tempthresh);
	cudaFree(d_tempbias);
	cudaFree(d_whchfea);
	cudaFree(d_clas);
	cudaFree(d_interresult);
	cudaFree(location);
	cudaFree(real_location);
	cudaFree(d_location);
	cudaFree(d_result);
	//free(h_x);
	free(testy);
	return t;
}

