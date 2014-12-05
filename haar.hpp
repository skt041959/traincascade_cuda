#ifndef HAAR_HPP_
#define HAAR_HPP_

#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace std;

typedef unsigned char u8;
typedef unsigned int u32;
typedef int s32;

typedef struct{
    int *pfeature;
    int featureNum;
}SFeature;

void checkCUDAError(const char *msg);

int integral_cuda(u8 *ptr, u8 *ptr_inte, int width, int height);

int calcuHaarFeature(u32 *ptr, vector<SFeature> &features, int width, int height);

#endif
