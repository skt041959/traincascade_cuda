#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//int prepare_sample(vector<vector<float> > &sample, vector<int> &flag);
int prepare_sample(float **sample, int **flag);
int prepare_test(float **sample, int **flag);
