#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//int prepare_sample(vector<vector<float> > &sample, vector<int> &flag);
int prepare_sample(float **sample, int **flag, int pos_num, int nag_num, bool write);
int prepare_test(float **sample, int **flag, int pos_num, int nag_num, bool write);

#define SAMPLE_COLS 19
#define SAMPLE_ROWS 19

