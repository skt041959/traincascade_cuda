#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "haar.hpp"

using namespace std;

__global__ void haar_edge_horizontal(u32 *ptr, s32 *pfeature, int width, int height, int sx, int sy)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int w = blockDim.x;

    int f1 = *(ptr+width*(i+sy)+(j+sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+sx));

    int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
        - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

    *(pfeature+w*i+j) = f1 - f2;
}

__global__ void haar_edge_vertical(u32 *ptr, s32 *pfeature, int width, int height, int sx, int sy)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int w = blockDim.x;

    int f1 = *(ptr+width*(i+sy)+(j+sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+sx));

    int f2 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

    *(pfeature+w*i+j) = f1 - f2;
}

__global__ void haar_liner_horizontal(u32 *ptr, s32 *pfeature, int width, int height, int sx, int sy)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int w = blockDim.x;

    int f1 = *(ptr+width*(i+sy)+(j+3*sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+3*sx));

    int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
        - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

    *(pfeature+w*i+j) = f1 - 2*f2;
}

__global__ void haar_liner_vertical(u32 *ptr, s32 *pfeature, int width, int height, int sx, int sy)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int w = blockDim.x;

    int f1 = *(ptr+width*(i+3*sy)+(j+sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+3*sy)+j) - *(ptr+width*i+(j+sx));

    int f2 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

    *(pfeature+w*i+j) = f1 - 2*f2;
}

__global__ void haar_rect(u32 *ptr, s32 *pfeature, int width, int height, int sx, int sy)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int w = blockDim.x;

    int f1 = *(ptr+width*(i+2*sy)+(j+2*sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*i+(j+2*sx));

    int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
        - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

    int f3 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

    *(pfeature+w*i+j) = f1 - 2*f2 -2*f3;
}

__host__ int calcuHaarFeature(u32 *ptr, vector<SFeature> &features, int width, int height)
{
    //for(int i=0; i<width; ++i)
    //{
    //    for(int j=0; j<height; ++j)
    //    {
    //        cout<<*(ptr+i*width+j)<<"\t";
    //    }
    //    cout<<endl;
    //}

    int memSize = (width+1)*(height+1)*sizeof(unsigned int);
    unsigned int *d_ptr;
    cudaMalloc((void **)&d_ptr, memSize);
    cudaMemcpy(d_ptr, ptr, memSize, cudaMemcpyHostToDevice);

    s32 *d_pfeature;
    int featureSize_max = width*height*sizeof(int);
    cudaMalloc((void **)&d_pfeature, featureSize_max);


    //=======================eage_horizontal==============================================
    int temp_x = 2;
    int temp_y = 1;
    int sx_max = width/temp_x;
    int sy_max = height/temp_y;
    for(int sx=1; sx<=sx_max; ++sx)
    {
        for(int sy=1; sy<=sy_max; ++sy)
        {
            int blockDim_x = width-temp_x*sx+1;
            int blockDim_y = height-temp_y*sy+1;

            dim3 dimGrid(1, 1);
            dim3 dimBlock(blockDim_x, blockDim_y);

            haar_edge_horizontal<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
            cudaThreadSynchronize();

            checkCUDAError("kernel execution");
            s32 *pfeature;
            int featureSize = blockDim_x*blockDim_y*sizeof(int);
            pfeature = (s32 *)malloc(featureSize);
            cudaMemcpy(pfeature, d_pfeature, featureSize, cudaMemcpyDeviceToHost);
            SFeature f;
            f.pfeature = pfeature;
            f.featureNum = featureSize;
            features.push_back(f);
        }
    }

    //=======================edge_vertical=====================================
    temp_x = 1;
    temp_y = 2;
    sx_max = width/temp_x;
    sy_max = height/temp_y;
    for(int sx=1; sx<=sx_max; ++sx)
    {
        for(int sy=1; sy<=sy_max; ++sy)
        {
            int blockDim_x = width-temp_x*sx+1;
            int blockDim_y = height-temp_y*sy+1;

            dim3 dimGrid(1, 1);
            dim3 dimBlock(blockDim_x, blockDim_y);

            haar_edge_vertical<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
            cudaThreadSynchronize();

            checkCUDAError("kernel execution");
            s32 *pfeature;
            int featureSize = blockDim_x*blockDim_y*sizeof(int);
            pfeature = (s32 *)malloc(featureSize);
            cudaMemcpy(pfeature, d_pfeature, featureSize, cudaMemcpyDeviceToHost);
            SFeature f;
            f.pfeature = pfeature;
            f.featureNum = featureSize;
            features.push_back(f);
        }
    }

    //======================liner_horizontal================================
    temp_x = 3;
    temp_y = 1;
    sx_max = width/temp_x;
    sy_max = height/temp_y;
    for(int sx=1; sx<=sx_max; ++sx)
    {
        for(int sy=1; sy<=sy_max; ++sy)
        {
            int blockDim_x = width-temp_x*sx+1;
            int blockDim_y = height-temp_y*sy+1;

            dim3 dimGrid(1, 1);
            dim3 dimBlock(blockDim_x, blockDim_y);

            haar_liner_horizontal<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
            cudaThreadSynchronize();

            checkCUDAError("kernel execution");
            s32 *pfeature;
            int featureSize = blockDim_x*blockDim_y*sizeof(int);
            pfeature = (s32 *)malloc(featureSize);
            cudaMemcpy(pfeature, d_pfeature, featureSize, cudaMemcpyDeviceToHost);
            SFeature f;
            f.pfeature = pfeature;
            f.featureNum = featureSize;
            features.push_back(f);
        }
    }

    //=====================liner_verical===================================
    temp_x = 1;
    temp_y = 3;
    sx_max = width/temp_x;
    sy_max = height/temp_y;
    for(int sx=1; sx<=sx_max; ++sx)
    {
        for(int sy=1; sy<=sy_max; ++sy)
        {
            int blockDim_x = width-temp_x*sx+1;
            int blockDim_y = height-temp_y*sy+1;

            dim3 dimGrid(1, 1);
            dim3 dimBlock(blockDim_x, blockDim_y);

            haar_liner_vertical<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
            cudaThreadSynchronize();

            checkCUDAError("kernel execution");
            s32 *pfeature;
            int featureSize = blockDim_x*blockDim_y*sizeof(int);
            pfeature = (s32 *)malloc(featureSize);
            cudaMemcpy(pfeature, d_pfeature, featureSize, cudaMemcpyDeviceToHost);
            SFeature f;
            f.pfeature = pfeature;
            f.featureNum = featureSize;
            features.push_back(f);
        }
    }

    //====================rect===============================================
    temp_x = 2;
    temp_y = 2;
    sx_max = width/temp_x;
    sy_max = height/temp_y;
    for(int sx=1; sx<=sx_max; ++sx)
    {
        for(int sy=1; sy<=sy_max; ++sy)
        {
            int blockDim_x = width-temp_x*sx+1;
            int blockDim_y = height-temp_y*sy+1;

            dim3 dimGrid(1, 1);
            dim3 dimBlock(blockDim_x, blockDim_y);

            haar_rect<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
            cudaThreadSynchronize();

            checkCUDAError("kernel execution");
            s32 *pfeature;
            int featureSize = blockDim_x*blockDim_y*sizeof(int);
            pfeature = (s32 *)malloc(featureSize);
            cudaMemcpy(pfeature, d_pfeature, featureSize, cudaMemcpyDeviceToHost);
            SFeature f;
            f.pfeature = pfeature;
            f.featureNum = featureSize;
            features.push_back(f);

            //cout<<blockDim_x<<" "<<blockDim_y<<endl;
            //for(int i=0; i<blockDim_y; ++i)
            //{
            //    cout<<"=";
            //    for(int j=0; j<blockDim_x; ++j)
            //    {
            //        cout<<*(pfeature+i*blockDim_x+j)<<"\t";
            //    }
            //    cout<<endl;
            //}
            //cout<<endl;
        }
    }

    cout<<endl;
    cudaFree(d_ptr);
    cudaFree(d_pfeature);

    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

