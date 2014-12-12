#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "haar.hpp"
#include "haar_kernels.cu"

using namespace std;
unsigned int *d_ptr; //image in the device
int memSize; //the image size in the device

float *d_pfeature; //feature in the device
int featureSize_max; //the compact feature size in the device

//int *d_offset_matrix; //the offset matrix of the feature store
//int offset_matrix_size_max;

float *p_features_start;
float *p_features[5]; //the individual feature of the 5type
int compactSize[5]; //the individual compact feature size of 5 type

int *p_offset_matrix[5]; //the individual offset of the 5type
int *d_offset_matrix[5];
int offset_matrix_size[5];

int temp_x[5] = {2, 1, 3, 1, 2};
int temp_y[5] = {1, 2, 1, 3, 2};

__host__ int prepare(float **p_raw_features, int *p_compactSize, int width, int height)
{
    *p_compactSize = 0;
    featureSize_max = 0;
    //offset_matrix_size_max = 0;

    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];

        offset_matrix_size[type] = sy_max*sx_max*width*height;

        //if(offset_matrix_size[type]>offset_matrix_size_max)
        //    offset_matrix_size_max = offset_matrix_size[type];

        p_offset_matrix[type] = (int*)malloc(offset_matrix_size[type]*sizeof(int));
        cudaMalloc((void **)(d_offset_matrix+type), offset_matrix_size[type]*sizeof(int));

        cout<<"malloc matrix "<<type<<endl;

        int *p_offset = p_offset_matrix[type];
        int index = 0;
        for(int sy=BASE_SCALE; sy<=sy_max; ++sy)
            for(int sx=BASE_SCALE; sx<=sx_max; ++sx)
            {
                //int blockDim_x = width-temp_x[type]*sx+1;
                //int blockDim_y = height-temp_y[type]*sy+1;
                for(int i=0; i<height; ++i)
                    for(int j=0; j<width; ++j)
                    {
                        if((i+temp_y[type]*sy) <= height && (j+temp_x[type]*sx) <= width)
                        {
                            *(p_offset+((sy-BASE_SCALE)*(sx_max-1)+(sx-BASE_SCALE))*width*height+width*i+j) = index;
                            //if(type==0)
                            //{
                            //    cout<<i<<","<<j<<" "<<sx<<","<<sy<<" ";
                            //    //cout<<((sy-1)*(sx_max-1)+(sx-1))*width*height+width*i+j<<endl;
                            //    cout<<index<<endl;
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

        cudaMemcpy(d_offset_matrix[type], p_offset_matrix[type], offset_matrix_size[type]*sizeof(int), cudaMemcpyHostToDevice);
    }


    cout<<"malloc feature"<<endl;
    p_features_start =  (float*)malloc(*p_compactSize*sizeof(float));
    *p_raw_features = p_features_start;

    float *t = p_features_start;
    p_features[0] = t;
    for(int type=1; type<5; ++type)
    {
        t += compactSize[type-1];
        p_features[type] = t;
    }

    memSize = (width+1)*(height+1);

    cudaMalloc((void **)&d_ptr, memSize*sizeof(unsigned int));
    cudaMalloc((void **)&d_pfeature, featureSize_max*sizeof(int));
    //cudaMalloc((void **)&d_offset_matrix, offset_matrix_size_max*sizeof(int));
    cout<<"memSize"<<memSize<<endl;
    cout<<"featureMax"<<featureSize_max<<endl;
    //cout<<"offsetMax"<<offset_matrix_size_max<<endl;

    return 0;
}

__host__ int calcuHaarFeature3(u32 *ptr, int width, int height)
{
    cudaMemcpy(d_ptr, ptr, memSize*sizeof(unsigned int), cudaMemcpyHostToDevice);
    //for(int i=0; i<20; ++i)
    //{
    //    for(int j=0; j<20; ++j)
    //        //cout<<i<<","<<j<<" "<<*(ptr+i*20+j)<<endl;
    //        cout<<*(ptr+i*20+j)<<" ";
    //    cout<<endl;
    //}

    for(int type=0; type<5; ++type)
    {
        int sx_max = width/temp_x[type];
        int sy_max = height/temp_y[type];

        dim3 dimGrid(sx_max-1, sy_max-1);
        dim3 dimBlock(width, height);
        switch(type)
        {
            case 0:
                haar_edge_horizontal3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix[0], d_pfeature, width+1, height+1);
                break;
            case 1:
                haar_edge_vertical3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix[1], d_pfeature, width+1, height+1);
                break;
            case 2:
                haar_liner_horizontal3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix[2], d_pfeature, width+1, height+1);
                break;
            case 3:
                haar_liner_vertical3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix[3], d_pfeature, width+1, height+1);
                break;
            case 4:
                haar_rect3<<<dimGrid, dimBlock>>>(d_ptr, d_offset_matrix[4], d_pfeature, width+1, height+1);
                break;
            default: break;
        }
        cudaThreadSynchronize();

        checkCUDAError("kernel execution");
        //cout<<"type "<<type<<endl;
        cudaMemcpy(p_features[type], d_pfeature, compactSize[type]*sizeof(float), cudaMemcpyDeviceToHost);
        //cout<<"======"<<endl;
        //if(type == 0)
        //{
        //    //for(int i=0; i<19; ++i)
        //    //    for(int j=0; j<19; ++j)
        //    //        cout<<i<<","<<j<<" "<<*(p_features[type]+i*19+j+100)<<endl;
        //    for(int i=0; i<compactSize[0]; ++i)
        //        cout<<i<<","<<*(p_features[0]+i)<<" "<<endl;
        //    cout<<endl;
        //}
    }

    return 0;
}

__host__ int post_calculate()
{
    cudaFree(d_ptr);
    cudaFree(d_pfeature);

    cudaFree(d_offset_matrix[0]);
    cudaFree(d_offset_matrix[1]);
    cudaFree(d_offset_matrix[2]);
    cudaFree(d_offset_matrix[3]);
    cudaFree(d_offset_matrix[4]);

    free(p_features_start);
    free(p_offset_matrix[0]);
    free(p_offset_matrix[1]);
    free(p_offset_matrix[2]);
    free(p_offset_matrix[3]);
    free(p_offset_matrix[4]);

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

