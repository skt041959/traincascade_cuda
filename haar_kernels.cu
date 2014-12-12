#include <cstdio>
#include <cstdlib>

#include "haar.hpp"

#define OFFSET 5

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

/*
__global__ void haar_tiled(u32 *ptr, s32 *pf1, s32 *pf2, s32 *pf3, s32 *pf4, s32 *pf5,
        int width, int height, int sx, int sy)
{
    __shared__ unsigned int data[m][n];
    int i = threadIdx.y;
    int j = threadIdx.x;
    int w = blockDim.x;

    int o = (blockIdx.y*OFFSET + threadIdx.y)*width + (blockIdx.x*OFFSET) + threadIdx.x;
    data[i][j] = *(ptr+o);
    __syncthreads();

    int f111 = *(ptr+width*(i+sy)+(j+sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+sx));

    int f121 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
        - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

    int f211 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

    int f133 = *(ptr+width*(i+sy)+(j+3*sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+3*sx));

    int f313 = *(ptr+width*(i+3*sy)+(j+sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+3*sy)+j) - *(ptr+width*i+(j+sx));

    int f224 = *(ptr+width*(i+2*sy)+(j+2*sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*i+(j+2*sx));

    *(pf1+w*i+j) = f1 - f2;

}
*/

__global__ void haar_edge_horizontal2(u32 *ptr, s32 *pfeature, int width, int height)
{
    unsigned int *info = (unsigned int*)pfeature;
    unsigned int i = (*(info+threadIdx.x) & 0xFF000000)>>24;
    unsigned int j = (*(info+threadIdx.x) & 0x00FF0000)>>16;
    unsigned int sx = (*(info+threadIdx.x) & 0x0000FF00)>>8;
    unsigned int sy = (*(info+threadIdx.x) & 0x000000FF)>>0;

    int f1 = *(ptr+width*(i+sy)+(j+sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+sx));

    int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
        - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

    *(pfeature+threadIdx.x) = f1 - f2;
}

__global__ void haar_edge_vertical2(u32 *ptr, s32 *pfeature, int width, int height)
{
    unsigned int *info = (unsigned int*)pfeature;
    unsigned int i = (*(info+threadIdx.x) & 0xFF000000)>>24;
    unsigned int j = (*(info+threadIdx.x) & 0x00FF0000)>>16;
    unsigned int sx = (*(info+threadIdx.x) & 0x0000FF00)>>8;
    unsigned int sy = (*(info+threadIdx.x) & 0x000000FF)>>0;

    int f1 = *(ptr+width*(i+sy)+(j+sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+sx));

    int f2 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

    *(pfeature+threadIdx.x) = f1 - f2;
}

__global__ void haar_liner_horizontal2(u32 *ptr, s32 *pfeature, int width, int height)
{
    unsigned int *info = (unsigned int*)pfeature;
    unsigned int i = (*(info+threadIdx.x) & 0xFF000000)>>24;
    unsigned int j = (*(info+threadIdx.x) & 0x00FF0000)>>16;
    unsigned int sx = (*(info+threadIdx.x) & 0x0000FF00)>>8;
    unsigned int sy = (*(info+threadIdx.x) & 0x000000FF)>>0;

    int f1 = *(ptr+width*(i+sy)+(j+3*sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+3*sx));

    int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
        - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

    *(pfeature+threadIdx.x) = f1 - 2*f2;
}

__global__ void haar_liner_vertical2(u32 *ptr, s32 *pfeature, int width, int height)
{
    unsigned int *info = (unsigned int*)pfeature;
    unsigned int i = (*(info+threadIdx.x) & 0xFF000000)>>24;
    unsigned int j = (*(info+threadIdx.x) & 0x00FF0000)>>16;
    unsigned int sx = (*(info+threadIdx.x) & 0x0000FF00)>>8;
    unsigned int sy = (*(info+threadIdx.x) & 0x000000FF)>>0;

    int f1 = *(ptr+width*(i+3*sy)+(j+sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+3*sy)+j) - *(ptr+width*i+(j+sx));

    int f2 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

    *(pfeature+threadIdx.x) = f1 - 2*f2;
}

__global__ void haar_rect2(u32 *ptr, s32 *pfeature, int width, int height)
{
    unsigned int *info = (unsigned int*)pfeature;
    unsigned int i = (*(info+threadIdx.x) & 0xFF000000)>>24;
    unsigned int j = (*(info+threadIdx.x) & 0x00FF0000)>>16;
    unsigned int sx = (*(info+threadIdx.x) & 0x0000FF00)>>8;
    unsigned int sy = (*(info+threadIdx.x) & 0x000000FF)>>0;

    int f1 = *(ptr+width*(i+2*sy)+(j+2*sx)) + *(ptr+width*i+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*i+(j+2*sx));

    int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
        - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

    int f3 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
        - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

    *(pfeature+threadIdx.x) = f1 - 2*f2 -2*f3;
}

__global__ void haar_edge_horizontal3(u32 *ptr, int *d_offset_matrix, float *pfeature, int width, int height)
{
    int i = threadIdx.y; //0
    int j = threadIdx.x; //13
    int sy = blockIdx.y+BASE_SCALE; //2
    int sx = blockIdx.x+BASE_SCALE; //3

    if((i+sy) <= (height-1) && (j+2*sx) <= (width-1))
    {
        int f1 = *(ptr+width*(i+sy)+(j+sx)) + *(ptr+width*i+j)
            - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+sx));

        int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
            - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

        int offset = *(d_offset_matrix+(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x*blockDim.y+blockDim.x*i+j);
        //int offset = (((sy-1)*gridDim.x+(sx-1))*blockDim.x*blockDim.y+blockDim.x*i+j);
        *(pfeature+offset) = f1-f2;
        //*(pfeature+offset) = offset;
    }
}

__global__ void haar_edge_vertical3(u32 *ptr, int *d_offset_matrix, float *pfeature, int width, int height)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int sx = blockIdx.x+BASE_SCALE;
    int sy = blockIdx.y+BASE_SCALE;

    if((i+2*sy) < (height-1) && (j+sx) < (width-1))
    {
        int f1 = *(ptr+width*(i+sy)+(j+sx)) + *(ptr+width*i+j)
            - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+sx));

        int f2 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
            - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

        int offset = *(d_offset_matrix+(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x*blockDim.y+blockDim.x*i+j);
        *(pfeature+offset) = f1-f2;
        //*(pfeature+offset) = f1;
        //*(pfeature+offset) = *(ptr+width*i+j);
    }
}

__global__ void haar_liner_horizontal3(u32 *ptr, int *d_offset_matrix, float *pfeature, int width, int height)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int sx = blockIdx.x+BASE_SCALE;
    int sy = blockIdx.y+BASE_SCALE;

    if((i+sy)<(height-1) && (j+3*sx)<(width-1))
    {
        int f1 = *(ptr+width*(i+sy)+(j+3*sx)) + *(ptr+width*i+j)
            - *(ptr+width*(i+sy)+j) - *(ptr+width*i+(j+3*sx));

        int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
            - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

        int offset = *(d_offset_matrix+(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x*blockDim.y+blockDim.x*i+j);
        *(pfeature+offset) = f1 -2*f2;
        //*(pfeature+offset) = f1;
        //*(pfeature+offset) = *(ptr+width*i+j);
    }
}

__global__ void haar_liner_vertical3(u32 *ptr, int *d_offset_matrix, float *pfeature, int width, int height)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int sx = blockIdx.x+BASE_SCALE;
    int sy = blockIdx.y+BASE_SCALE;

    if((i+3*sy)<(height-1) && (j+sx)<(width-1))
    {
        int f1 = *(ptr+width*(i+3*sy)+(j+sx)) + *(ptr+width*i+j)
            - *(ptr+width*(i+3*sy)+j) - *(ptr+width*i+(j+sx));

        int f2 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
            - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

        int offset = *(d_offset_matrix+(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x*blockDim.y+blockDim.x*i+j);
        *(pfeature+offset) = f1-2*f2;
        //*(pfeature+offset) = f1;
        //*(pfeature+offset) = *(ptr+width*i+j);
    }
}

__global__ void haar_rect3(u32 *ptr, int *d_offset_matrix, float *pfeature, int width, int height)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int sx = blockIdx.x+BASE_SCALE;
    int sy = blockIdx.y+BASE_SCALE;

    if((i+2*sy)<(height-1) && (j+2*sx)<(width-1))
    {
        int f1 = *(ptr+width*(i+2*sy)+(j+2*sx)) + *(ptr+width*i+j)
            - *(ptr+width*(i+2*sy)+j) - *(ptr+width*i+(j+2*sx));

        int f2 = *(ptr+width*(i+sy)+(j+2*sx)) + *(ptr+width*i+(j+sx))
            - *(ptr+width*(i+sy)+(j+sx)) - *(ptr+width*i+(j+2*sx));

        int f3 = *(ptr+width*(i+2*sy)+(j+sx)) + *(ptr+width*(i+sy)+j)
            - *(ptr+width*(i+2*sy)+j) - *(ptr+width*(i+sy)+(j+sx));

        int offset = *(d_offset_matrix+(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x*blockDim.y+blockDim.x*i+j);
        *(pfeature+offset) = f1-2*f2-2*f3;
        //*(pfeature+offset) = f1;
        //*(pfeature+offset) = *(ptr+width*i+j);
    }
}

