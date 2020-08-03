#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <bitset>
#include <vector>
#include <ctime>

using namespace std;
using namespace cv;
/*
!!! It needs to chagne bitest template value to (num_patch * num_patch)C2 * 2, when you want to change num_patch !!!
*/
#define BITSET_LENGTH 240
/*
Function description
This function divides each frames into num_patch X num_patch partition.
Each parition get sum of pixels they have.
After that compute the ternary digit for every possible pair of patches to obtain the frame descriptor, where N = {(1, 2), (1, 3), ...}
Function store tenary digit in each two bit in int type variable. if digit exceed integer scale store digits in another integer.
All integer are wrapped in python list.

*/
void tenary_vectorzie(bitset<BITSET_LENGTH> &result, const Mat frame, int num_patch = 4, int threshold = 20) {
    vector<int> ternary(num_patch*num_patch, 0);
    int width = frame.size().width;
    int height = frame.size().height;
    uchar *pixel = frame.data;
    int partition_width = width / num_patch;
    int partition_height = height / num_patch;
    result.reset();
    // Dvide frame into num_patch*num_patch parts and store each sum of parts into vector array
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j) {
            ternary[num_patch*(i / partition_height) + j / partition_width ] += pixel[i*width*3 + j*3] + pixel[i*width*3 + j*3 + 1] + pixel[i*width*3 + j*3 + 2];
        }

    // Compare all pairs of list which stores sum of divided frame and store result into bitset.
    int num = 0;
    for (int i = 0; i < num_patch*num_patch; ++i)
        for (int j = i+1; j < num_patch*num_patch; ++j) {
            if (ternary[i] > ternary[j] + threshold)
                result[num] = result[num+1] = 1;
            else if (ternary[i] >= ternary[j] - threshold)
                result[num+1] = 1;
            num += 2;
        }
}
/*
Argument have to be list type.
Function computes sum of each 2 digits in integer array.
In other words, function divieds all integers in list into 2 digits, sum all of them.
*/
int bit_sum(const bitset<BITSET_LENGTH> bits) {
    int result = 0;

    for (int i = 0; i < bits.size(); i += 2) {
        result += bits[i] * 2 + bits[i+1];
    }

    return result;
}

int main() {
    Mat h_frame, f_frame;
    VideoCapture game("/home/lol/20200701_T1_DWG_GEN_SB.mp4");
    VideoCapture highlight("/home/lol/20200701_hl.mp4");
    bitset<BITSET_LENGTH> h_bits, f_bits;

    if (!game.isOpened()) {
        printf("Fullgame file can not open\n");
        return 1;
    }
    if (!highlight.isOpened()) {
        printf("Highlight file can not open\n");
        return 1;
    }

    game.set(CAP_PROP_POS_FRAMES, 130250);
    highlight.set(CAP_PROP_POS_FRAMES, 5995);
    double fps_game = game.get(CAP_PROP_FPS);
    double fps_highlight = highlight.get(CAP_PROP_FPS);

    highlight.read(h_frame);
    int f_num = 130250;
    tenary_vectorzie(h_bits, h_frame);
    
    clock_t start_time = clock();
    int sim;
    while (f_num < 148250) {
        game.read(f_frame);
        tenary_vectorzie(f_bits, f_frame);
 
        sim = bit_sum(f_bits^h_bits);
        if (sim < 10)
            printf("find! f_num : %lf => %d\n", game.get(CAP_PROP_POS_FRAMES), sim);
        f_num++;
    }
    int finish_time = int(clock() - start_time);
    printf("%ld.%ldsec\n", finish_time / CLOCKS_PER_SEC, finish_time % CLOCKS_PER_SEC);

    return 0;
}