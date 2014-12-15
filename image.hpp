#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class Tile
{
    public:
        Tile(int x, int y, int width, int height):
            x(x),
            y(y),
            width(width),
            height(height)
        {
            x2 = width+x;
            y2 = height+y;
        }
        Tile(const Tile & t)
        {
            x = t.x;
            y = t.y;
            x2 = t.x2;
            y2 = t.y2;
            width = t.width;
            height = t.height;
        }
        int x;
        int y;
        int width;
        int height;
        int x2;
        int y2;
};


//int prepare_sample(vector<vector<float> > &sample, vector<int> &flag);
int prepare_sample(float **sample, int **flag, int pos_num, int nag_num, bool write);
int prepare_test(float **sample, int **flag, int pos_num, int nag_num, bool write);
int prepare_image(char * filename, float **feature, vector<Tile> &place);
int show_image(char *filename, vector<Tile> &faces);

#define SAMPLE_COLS 19
#define SAMPLE_ROWS 19

