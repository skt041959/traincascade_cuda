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

typedef enum{
    EDGE_H = 0,
    EDGE_V = 1,
    LINER_H = 2,
    LINER_V = 3,
    RECT = 4
}TemplateType;

void checkCUDAError(const char *msg);

int integral_cuda(u8 *ptr, u8 *ptr_inte, int width, int height);

int calcuHaarFeature(u32 *ptr, vector<SFeature> &features, int width, int height);
int calcuHaarFeature_tiled(u32 *ptr, SFeature *features, int width, int height);

#endif
