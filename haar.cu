#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "haar.hpp"
#include "haar_kernels.cu"

using namespace std;

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

    int temp_x[5] = {2, 1, 3, 1, 2};
    int temp_y[5] = {1, 2, 1, 3, 2};
    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];

        for(int sx=1; sx<=sx_max; ++sx)
        {
            for(int sy=1; sy<=sy_max; ++sy)
            {
                int blockDim_x = width-temp_x[type]*sx+1;
                int blockDim_y = height-temp_y[type]*sy+1;

                dim3 dimGrid(1, 1);
                dim3 dimBlock(blockDim_x, blockDim_y);

                switch(type)
                {
                case EDGE_H:
                    haar_edge_horizontal<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                    break;
                case EDGE_V:
                    haar_edge_vertical<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                    break;
                case LINER_H:
                    haar_liner_horizontal<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                    break;
                case LINER_V:
                    haar_liner_vertical<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                    break;
                case RECT:
                    haar_rect<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                    break;
                default: break;
                }
                cudaThreadSynchronize();

                checkCUDAError("kernel execution");
                s32 *pfeature;
                int featureSize = blockDim_x*blockDim_y*sizeof(int);
                pfeature = (s32 *)malloc(featureSize);
                cudaMemcpy(pfeature, d_pfeature, featureSize, cudaMemcpyDeviceToHost);
                SFeature f;
                f.pfeature = pfeature;
                f.featureNum = blockDim_x*blockDim_y;
                features.push_back(f);
            }
        }
    }

    cout<<endl;
    cudaFree(d_ptr);
    cudaFree(d_pfeature);

    return 0;
}

__host__ int prepare(s32 **p_raw_features, int **compressSize, unsigned int **d_ptr, int **d_pfeature, int width, int height)
{
    int s1 = (width)*(height/2)*width*height*sizeof(int);
    int s2 = (width/2)*(height)*width*height*sizeof(int);
    int featureSize_max = s1 > s2 ? s1 : s2;

    int * m = (int *)malloc(5*featureSize_max*sizeof(int));
    *p_raw_features = m;

    int temp_x[5] = {2, 1, 3, 1, 2};
    int temp_y[5] = {1, 2, 1, 3, 2};
    int total[5] = {0};
    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];

        for(int sx=1; sx<=sx_max; ++sx)
        {
            for(int sy=1; sy<=sy_max; ++sy)
            {
                int blockDim_x = width-temp_x[type]*sx+1;
                int blockDim_y = height-temp_y[type]*sy+1;
                total[type] += blockDim_x*blockDim_y;
            }
        }
    }
    *compressSize = total;

    int memSize = (width+1)*(height+1)*sizeof(unsigned int);
    cudaMalloc((void **)d_ptr, memSize);

    cudaMalloc((void **)d_pfeature, featureSize_max);

}

__host__ int calcuHaarFeature3(u32 *ptr, s32 *p_raw_features, int width, int height, )
{
    int s1 = (width)*(height/2)*width*height*sizeof(int);
    int s2 = (width/2)*(height)*width*height*sizeof(int);
    int featureSize_max = s1 > s2 ? s1 : s2;

    int * pfeatures[5];
    pfeatures[0] = p_raw_features;
    pfeatures[1] = p_raw_features+featureSize_max;
    pfeatures[2] = p_raw_features+2*featureSize_max;
    pfeatures[3] = p_raw_features+3*featureSize_max;
    pfeatures[4] = p_raw_features+4*featureSize_max;

    int memSize = (width+1)*(height+1)*sizeof(unsigned int);
    cudaMemcpy(d_ptr, ptr, memSize, cudaMemcpyHostToDevice);

    int temp_x[5] = {2, 1, 3, 1, 2};
    int temp_y[5] = {1, 2, 1, 3, 2};

    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];

        dim3 dimGrid(sy_max, sx_max);
        dim3 dimBlock(width, height);

        switch(type)
        {
            case EDGE_H:
                haar_edge_horizontal3<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                break;
            case EDGE_V:
                haar_edge_vertical3<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                break;
            case LINER_H:
                haar_liner_horizontal3<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                break;
            case LINER_V:
                haar_liner_vertical3<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                break;
            case RECT:
                haar_rect3<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1, sx, sy);
                break;
            default: break;
        }
        cudaThreadSynchronize();

        checkCUDAError("kernel execution");
        int featureSize = blockDim_x*blockDim_y*sizeof(int);
        cudaMemcpy(pfeatures[type], d_pfeature, featureSize, cudaMemcpyDeviceToHost);
    }

    cout<<endl;
    cudaFree(d_ptr);
    cudaFree(d_pfeature);

    return 0;

}

//obsolete
__host__ int calcuHaarFeature2(u32 *ptr, SFeature *features, int width, int height)
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

    int temp_x[5] = {2, 1, 3, 1, 2};
    int temp_y[5] = {1, 2, 1, 3, 2};

    int total[5] = {0};

    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];

        for(int sx=1; sx<=sx_max; ++sx)
        {
            for(int sy=1; sy<=sy_max; ++sy)
            {
                int blockDim_x = width-temp_x[type]*sx+1;
                int blockDim_y = height-temp_y[type]*sy+1;
                total[type] += blockDim_x*blockDim_y;
            }
        }
        printf("total %d\n", total[type]);
    }

    unsigned int *total_index[5];
    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];
        unsigned int *index = (unsigned int*)malloc(total[type]*sizeof(unsigned int));
        total_index[type] = index;

        for(unsigned char sx=1; sx<=sx_max; ++sx)
        {
            for(unsigned char sy=1; sy<=sy_max; ++sy)
            {
                unsigned char blockDim_x = width-temp_x[type]*sx+1;
                unsigned char blockDim_y = height-temp_y[type]*sy+1;

                for(unsigned char i=0; i<blockDim_y; ++i)
                {
                    for(unsigned char j=0; j<blockDim_x; ++j)
                    {
                        *index = ( i<<24) | ( j<<16) | ( sy<<8) | ( sx);
                        index++;
                    }
                }
            }
        }
    }

    unsigned int *d_offset_scale;
    cudaMalloc((void **)&d_offset_scale, total[0]*sizeof(unsigned int));
    s32 *d_pfeature;
    cudaMalloc((void **)&d_pfeature, total[0]*sizeof(unsigned int));

    printf("1\n");

    for(int type=0; type<5; ++type)
    {
        dim3 dimGrid(1, 1);
        dim3 dimBlock(total[type], 1);
        cudaMemcpy(d_offset_scale, total_index[type], total[type]*sizeof(unsigned int), cudaMemcpyHostToDevice);

        switch(type)
        {
        case EDGE_H:
            haar_edge_horizontal2<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1);
            break;
        case EDGE_V:
            haar_edge_vertical2<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1);
            break;
        case LINER_H:
            haar_liner_horizontal2<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1);
            break;
        case LINER_V:
            haar_liner_vertical2<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1);
            break;
        case RECT:
            haar_rect2<<<dimGrid, dimBlock>>>(d_ptr, d_pfeature, width+1, height+1);
            break;
        default: break;
        }
        cudaThreadSynchronize();

        checkCUDAError("kernel execution");

        s32 *pfeature  = (s32 *)malloc(total[type]*sizeof(int));
        printf("2\n");

        cudaMemcpy(pfeature, d_pfeature, total[type]*sizeof(int), cudaMemcpyDeviceToHost);
        (features+type)->pfeature = pfeature;
        (features+type)->featureNum = total[type];
        printf("%d\n", total[type]);
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

