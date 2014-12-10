#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "haar.hpp"
#include "haar_kernels.cu"

using namespace std;
unsigned int *d_ptr; //image in the device
int memSize; //the image size in the device

int *d_pfeature; //feature in the device
int featureSize_max; //the compact feature size in the device

int *d_offset_matrix; //the offset matrix of the feature store
int offset_matrix_size_max;

int *p_features_start;
int *p_features[5]; //the individual feature of the 5type
int compactSize[5]; //the individual compact feature size of 5 type

int *p_offset_matrix[5]; //the individual offset of the 5type
int offset_matrix_size[5];

int temp_x[5] = {2, 1, 3, 1, 2};
int temp_y[5] = {1, 2, 1, 3, 2};


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

    //cout<<endl;
    cudaFree(d_ptr);
    cudaFree(d_pfeature);

    return 0;
}

__host__ int prepare(s32 **p_raw_features, int *p_compactSize, int width, int height)
{
    *p_compactSize = 0;
    featureSize_max = 0;
    offset_matrix_size_max = 0;

    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];

        offset_matrix_size[type] = sy_max*sx_max*width*height;

        if(offset_matrix_size[type]>offset_matrix_size_max)
            offset_matrix_size_max = offset_matrix_size[type];

        p_offset_matrix[type] = (int*)malloc(offset_matrix_size[type]*sizeof(int));
        cout<<"malloc matrix "<<type<<endl;

        int *p_offset = p_offset_matrix[type];
        int index = 0;
        for(int sy=1; sy<=sy_max; ++sy)
            for(int sx=1; sx<=sx_max; ++sx)
            {
                //int blockDim_x = width-temp_x[type]*sx+1;
                //int blockDim_y = height-temp_y[type]*sy+1;
                for(int i=0; i<height; ++i)
                    for(int j=0; j<width; ++j)
                    {
                        if((i+temp_y[type]*sy) <= height && (j+temp_x[type]*sx) <= width)
                        {
                            *(p_offset+((sy-1)*(sx_max-1)+(sx-1))*width*height+width*i+j) = index;
                            //if(type==0 && sy==2 && sx==1)
                            //{
                            //    cout<<i<<","<<j<<" ";
                            //    cout<<((sy-1)*(sx_max-1)+(sx-1))*width*height+width*i+j<<endl;
                            //}
                            index++;
                        }
                    }
            }

        cout<<"------"<<index<<endl;

        compactSize[type] = index;
        *p_compactSize += compactSize[type];

        if(index>featureSize_max)
            featureSize_max = index;
    }

    cout<<"malloc feature"<<endl;
    p_features_start =  (int*)malloc(*p_compactSize*sizeof(int));
    *p_raw_features = p_features_start;

    int *t = p_features_start;
    p_features[0] = t;
    for(int type=1; type<5; ++type)
    {
        t += compactSize[type-1];
        p_features[type] = t;
    }

    int memSize = (width+1)*(height+1)*sizeof(unsigned int);

    cudaMalloc((void **)&d_ptr, memSize);
    cudaMalloc((void **)&d_pfeature, featureSize_max*sizeof(int));
    cudaMalloc((void **)&d_offset_matrix, offset_matrix_size_max*sizeof(int));
    cout<<"memSize"<<memSize<<endl;
    cout<<"featureMax"<<featureSize_max<<endl;
    cout<<"offsetMax"<<offset_matrix_size_max<<endl;

    return 0;
}

__host__ int calcuHaarFeature3(u32 *ptr, int width, int height)
{
    cudaMemcpy(d_ptr, ptr, memSize, cudaMemcpyHostToDevice);

    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];

        dim3 dimGrid(sx_max-1, sy_max-1);
        dim3 dimBlock(width, height);

        cudaMemcpy(d_offset_matrix, p_offset_matrix[type], offset_matrix_size[type]*sizeof(int), cudaMemcpyHostToDevice);

        switch(type)
        {
            case EDGE_H:
                haar_edge_horizontal3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix, d_pfeature, width+1, height+1);
                break;
            case EDGE_V:
                haar_edge_vertical3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix, d_pfeature, width+1, height+1);
                break;
            case LINER_H:
                haar_liner_horizontal3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix, d_pfeature, width+1, height+1);
                break;
            case LINER_V:
                haar_liner_vertical3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix, d_pfeature, width+1, height+1);
                break;
            case RECT:
                haar_rect3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix, d_pfeature, width+1, height+1);
                break;
            default: break;
        }
        cudaThreadSynchronize();

        checkCUDAError("kernel execution");
        cout<<"type "<<type<<endl;
        cudaMemcpy(p_features[type], d_pfeature, compactSize[type]*sizeof(int), cudaMemcpyDeviceToHost);
        //cout<<"======"<<endl;
        //for(int i=0; i<20; ++i)
        //    for(int j=0; j<20; ++j)
        //        cout<<i<<","<<j<<" "<<*(p_features[type]+i*20+j)<<endl;
    }

    return 0;
}

__host__ int post_calculate()
{
    cudaFree(d_ptr);
    cudaFree(d_pfeature);
    cudaFree(d_offset_matrix);

    free(p_features_start);
    free(p_offset_matrix[0]);
    free(p_offset_matrix[1]);
    free(p_offset_matrix[2]);
    free(p_offset_matrix[3]);
    free(p_offset_matrix[4]);

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

    //cout<<endl;
    cudaFree(d_ptr);
    cudaFree(d_pfeature);

    return 0;
}

__host__ void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

