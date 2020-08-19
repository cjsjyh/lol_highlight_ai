#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <bitset>
#include <vector>
#include <ctime>
#include <fstream>
#include <utility>
#include <algorithm>
#include <queue>
#include <map>
#include <thread>
#include <stdexcept>
#include <mutex>

using namespace std;
using namespace cv;
// !!! It needs to chagne bitest template value to (num_patch * num_patch)C2 * 2, when you want to change num_patch !!!
#define BITSET_LENGTH 600 
#define THRESHOLD_VALUE 30
// first value is zero when not finding same frame between full game and highlight yet. If not, It it value of number of full frame.
// second value is number of highlight frame. 
pair<int, int> thread_framematch_result;
string highligh_name, fullgame_name;
mutex mutex_framereader, mutex_frame; // mutex for thread_normal_find thread.
int framereadcount = 0; // value for mutex.
/*
Function description
This function divides each frames into num_patch X num_patch partition.
Each parition get sum of pixels they have.
After that compute the ternary digit for every possible pair of patches to obtain the frame descriptor, where N = {(1, 2), (1, 3), ...}
Function store tenary digit in each two bit in int type variable. if digit exceed integer scale store digits in another integer.
All integer are wrapped in python list.
*/
void tenary_vectorzie(bitset<BITSET_LENGTH> &result, const Mat frame, int num_patch = 5, int threshold = 20);
/*
Argument have to be list type.
Function computes sum of each 2 digits in integer array.
In other words, function divieds all integers in list into 2 digits, sum all of them.
*/
int bit_sum(const bitset<BITSET_LENGTH> bits);
bool sim_comp(pair<int, int> a, pair<int, int> b) {return a.second < b.second;}
int frame_search(const deque<pair<int, Mat>> &frames, int pos);
/*
It is for thread function.
It find same frame between full game and highlight.
If fail to find same frame for 2000 frames, It creates same thread with itself.
If one of threads finds same frame, all thread is destroyed. and threads which is created eariler find same frame to found frame num.
*/
void thread_normal_find (map<int, Mat> &frame_map, int pos_game, int pos_highlight);

int main() {
    enum FIND_STATE {NORMAL_FIND, FAST_FIND, SLOW_FIND};
    Mat h_frame, f_frame, copy_frame, temp_frame;
    VideoCapture game(fullgame_name);
    VideoCapture highlight(highligh_name);
    bitset<BITSET_LENGTH> h_bits, f_bits;

    if (!game.isOpened()) {
        printf("Fullgame file can not open\n");
        return 1;
    }
    if (!highlight.isOpened()) {
        printf("Highlight file can not open\n");
        return 1;
    }
    
    // highlight.set(CAP_PROP_POS_FRAMES, 5570);
    // game.set(CAP_PROP_POS_FRAMES, 42014);
    // highlight.read(h_frame);
    // tenary_vectorzie(h_bits, h_frame);
    // game.read(f_frame);
    // tenary_vectorzie(f_bits, f_frame);
    // cout << highlight.get(CAP_PROP_POS_FRAMES) << " : " << game.get(CAP_PROP_POS_FRAMES) << " => " << bit_sum(h_bits^f_bits) << '\n';
    // cout << h_bits << '\n';
    // cout << f_bits << '\n';
    // return 0;

    int last_highlight = highlight.get(CAP_PROP_FRAME_COUNT);
    int last_game = game.get(CAP_PROP_FRAME_COUNT);
    clock_t start_time = clock();
    highlight.set(CAP_PROP_POS_FRAMES, 1100);
    game.set(CAP_PROP_POS_FRAMES, 0);
    int pos_highlight = 1100, pos_game = 0, matched_pos_game;
    double fps_game = game.get(CAP_PROP_FPS);
    double fps_highlight = highlight.get(CAP_PROP_FPS);
    int sim;
    ofstream file("output.txt");
    int state = NORMAL_FIND;
    vector<pair<int, int>> sims; // left->idx, right->simillarity
    deque<pair<int, Mat>> frames;
    int idx;
    int scene_count = 0;

    file << "f.mp4" << '\n';

    highlight.read(h_frame);
    while (pos_highlight < last_highlight) {
        cout << "highlight postion : " << pos_highlight << '\n';
        tenary_vectorzie(h_bits, h_frame);
        while (pos_game < last_game) { // find similar frame per each frame. This case is finding first game frame similar to highligh frame
            if (pos_game % 1000 == 0) {
                cout << "game postion : " << pos_game << '\n';
            }
            if (state == NORMAL_FIND) {
                idx = frame_search(frames, pos_game);
                if (idx != -1 && idx != frames.size()) {
                    cout << "NORMAL FIND start : " << pos_game << '/' << idx << '\n';
                    f_frame = frames[idx].second;
                }
                else {
                    // cout << "NORMAL FIND : There are no frame in deq. First idx of deq : " << frames[0].first << '\n';
                    game.read(f_frame);
                    f_frame.copyTo(copy_frame);
                    temp_frame = copy_frame.clone();
                    frames.push_back(make_pair(pos_game, temp_frame));
                }
                tenary_vectorzie(f_bits, f_frame);
                sim = bit_sum(h_bits^f_bits);
                if (sim < THRESHOLD_VALUE) {
                    sims.push_back(make_pair(pos_game, sim));
                }
                else if (!sims.empty() && sim >= 40) {
                    sort(sims.begin(), sims.end(), sim_comp);
                    cout << pos_highlight << " / " << sims[0].first << "=>" << sims[0].second << " NORMAL STATE" << '\n';
                    // file << "start " << pos_highlight << " / " << sims[0].first << " => " << sims[0].second << '\n';
                    file << sims[0].first;
                    state = FAST_FIND;
                    pos_game = sims[0].first+1;
                    while (!frames.empty() && frames[0].first < pos_game)
                        frames.pop_front();
                    sims.clear();
                    break;
                }
                pos_game++;
            }
            else if (state == FAST_FIND) { // find similar frame per one second
                if (frames.empty())
                    cout << pos_game << " / " << game.get(CAP_PROP_POS_FRAMES) << " / " << "empty\n";
                else
                    cout << pos_game << " / " << game.get(CAP_PROP_POS_FRAMES) << " / " << frames.back().first << '\n';
                idx = pos_game + (int)fps_game - 5;
                while (pos_game < idx) {
                    if (frames.empty() || frames.back().first < pos_game) {
                        game.read(f_frame);
                        f_frame.copyTo(copy_frame);
                        temp_frame = copy_frame.clone();
                        frames.push_back(make_pair(pos_game, temp_frame));
                    }
                    pos_game++;
                }
                idx = frame_search(frames, pos_game);
                for (int i = 0;  i< 10; ++i) {
                    if (idx != -1 && idx != frames.size()) {
                        f_frame = frames[idx++].second;
                    }
                    else {
                        game.read(f_frame);
                        f_frame.copyTo(copy_frame);
                        temp_frame = copy_frame.clone();
                        frames.push_back(make_pair(pos_game, temp_frame));
                    }
                    tenary_vectorzie(f_bits, f_frame);
                    sim = bit_sum(h_bits^f_bits);
                    sims.push_back(make_pair(pos_game, sim));
                    pos_game++;
                }
                sort(sims.begin(), sims.end(), sim_comp);
                if (sims[0].second > THRESHOLD_VALUE) { //fail to find similar frame in next second frame of previous one.
                    if (sims[0].second >= 100) { // Case where move to next highlight frame scene.
                        highlight.set(CAP_PROP_POS_FRAMES, highlight.get(CAP_PROP_POS_FRAMES) - (int)fps_highlight+1);
                        pos_highlight -= ((int)fps_highlight-1);
                        pos_game -= ((int)fps_game + 5);
                        idx = frame_search(frames, pos_game);
                        cout << highlight.get(CAP_PROP_POS_FRAMES) << " : " << pos_highlight << " / " << game.get(CAP_PROP_POS_FRAMES) << " : " << pos_game << " / " << idx << " / " << frames.size() << '\n';
                        do {
                            highlight.read(h_frame);
                            tenary_vectorzie(h_bits, h_frame);
                            if (idx != -1 && idx != frames.size()) {
                                cout << "zzloozz " << frames[idx].first << '\n';
                                f_frame = frames[idx++].second;
                            }
                            else {
                                cout << "wtf\n";
                                game.read(f_frame);
                                f_frame.copyTo(copy_frame);
                                temp_frame = copy_frame.clone();
                                frames.push_back(make_pair(pos_game, temp_frame));
                            }
         
                            tenary_vectorzie(f_bits, f_frame);
                            sim = bit_sum(h_bits^f_bits);
                            cout << "next highlight " << pos_highlight << " / " << pos_game << " => " << sim << '\n';
                            pos_highlight++; pos_game++;
                        } while (sim <= (int)(THRESHOLD_VALUE * 1.5));
                        state = NORMAL_FIND;
                        cout << pos_highlight << " / " << pos_game << " move to next highlight frame scene\n";
                        // file << "end " << pos_highlight << " / " << pos_game << '\n';
                        file << " " << pos_game << '\n';
                        for (int i = 0; i < 25; ++i) {
                            highlight.read(h_frame);
                            pos_highlight++;
                        }
                        pos_game = frames[0].first;
                    }
                    else { // Case where just fail to find similar frame in same scene.
                        pos_game -= 5;
                        cout << "fail to find similar frame in same scene\n";
                    }
                }
                else { // find similar frame in next second frame of previous one.
                    cout << pos_highlight << " : " << highlight.get(CAP_PROP_POS_FRAMES) << " / " << sims[0].first << "=>" << sims[0].second << " FAST STATE" <<'\n';
                    // file << pos_highlight << " / " << sims[0].first << '\n';
                    pos_game = sims[0].first+1;
                    while (!frames.empty() && frames[0].first < pos_game)
                        frames.pop_front();
                    cout << "delete " << pos_game-1 << '\n';
                }
                sims.clear();
                break;
            }
        }
        if (state == FAST_FIND) {
            for (int i  =0; i < (int)fps_highlight; ++i) {
                highlight.read(h_frame);
                ++pos_highlight;
            }
        }
        else if (state == NORMAL_FIND) {
            highlight.read(h_frame);
            pos_highlight++;
        }
    }
    int finish_time = int(clock() - start_time);
    printf("%ld.%ldsec\n", finish_time / CLOCKS_PER_SEC, finish_time % CLOCKS_PER_SEC);
    file.close();

    return 0;
}

/*
Function description
This function divides each frames into num_patch X num_patch partition.
Each parition get sum of pixels they have.
After that compute the ternary digit for every possible pair of patches to obtain the frame descriptor, where N = {(1, 2), (1, 3), ...}
Function store tenary digit in each two bit in int type variable. if digit exceed integer scale store digits in another integer.
All integer are wrapped in python list.
*/
void tenary_vectorzie(bitset<BITSET_LENGTH> &result, const Mat frame, int num_patch, int threshold) {
    vector<int> ternary(num_patch*num_patch, 0);
    int width = frame.size().width;
    int height = frame.size().height;
    uchar *pixel = frame.data;
    int partition_width = width / num_patch;
    int partition_height = height / num_patch;
    threshold *= partition_width * partition_height;

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
Function computes sum of each 2 digits in integer array and return sum.
In other words, function divieds all integers in list into 2 digits, sum all of them.
*/
int bit_sum(const bitset<BITSET_LENGTH> bits) {
    int result = 0;

    for (int i = 0; i < bits.size(); i += 2) {
        result += bits[i] * 2 + bits[i+1];
    }

    return result;
}

int frame_search(const deque<pair<int, Mat>> &frames, int pos) {
    if (frames.empty() || frames[0].first > pos || frames.back().first < pos) {
        return -1;
    }
    
    int left = 0;
    int right = (int)frames.size()-1;
    int mid;

    while (left <= right) {
        mid = (right + left) / 2;
        if (frames[mid].first < pos)
            left = mid + 1;
        else if (pos < frames[mid].first)
            right = mid -1;
        else
            return mid;
    }

    return -1;
}
/*
It is for thread function.
It find same frame between full game and highlight.
If fail to find same frame for 2000 frames, It creates same thread with itself.
If one of threads finds same frame, all thread is destroyed. and threads which is created eariler find same frame to found frame num.
function returns true if finished correctly, otherwise returns false.
*/
void thread_normal_find (map<int, Mat> &frame_map, int pos_game, int pos_highlight) {
    VideoCapture game(fullgame_name);
    VideoCapture highlight(highligh_name);
    bitset<BITSET_LENGTH> h_bits, f_bits;
    int pos_game_origin = pos_game;
    Mat h_frame, f_frame, copy_frame, temp_frame;
    int last_game = game.get(CAP_PROP_FRAME_COUNT);
    int last_highlight = highlight.get(CAP_PROP_FRAME_COUNT);
    double fps_highlight = highlight.get(CAP_PROP_FPS);
    highlight.set(CAP_PROP_POS_FRAMES, pos_highlight);
    game.set(CAP_PROP_POS_FRAMES, pos_game);
    int sim;
    vector<pair<int, int>> sims; // left->idx, right->simillarity

    highlight.read(h_frame);
    tenary_vectorzie(h_bits, h_frame);
    while (thread_framematch_result.first == 0 && pos_game < last_game) {
        try {
            mutex_framereader.lock();
            framereadcount++;
            if (framereadcount == 1) mutex_frame.lock();
            mutex_framereader.unlock();
            f_frame = frame_map.at(pos_game);
            mutex_framereader.lock();
            framereadcount--;
            if (framereadcount == 0) mutex_frame.unlock();
            mutex_framereader.unlock();
        } catch(const out_of_range& oor) {
            if (pos_game == game.get(CAP_PROP_POS_FRAMES)) {

            }
            game.read(f_frame);
            f_frame.copyTo(copy_frame);
            temp_frame = copy_frame.clone();
            mutex_frame.lock();
            frame_map.insert({pos_game, temp_frame});
            mutex_frame.unlock();
        }

        if (pos_game == pos_game_origin + 2000 && last_highlight > pos_highlight + (int)fps_highlight) {
            thread t1(thread_normal_find, ref(frame_map), pos_game_origin, pos_highlight + (int)fps_highlight);
            t1.join();
        }
        tenary_vectorzie(f_bits, f_frame);
        sim = bit_sum(h_bits^f_bits);
        if (sim < THRESHOLD_VALUE) {
            sims.push_back(make_pair(pos_game, sim));
        }
        else if (!sims.empty() && sim >= 40) {
            sort(sims.begin(), sims.end(), sim_comp);
            mutex_framereader.lock();
            if (thread_framematch_result.second > pos_highlight)
                thread_framematch_result = {sims[0].first, pos_highlight};
            mutex_framereader.unlock();
            break;
        }
        pos_game++;
    }

    if (thread_framematch_result.second > pos_highlight) {
        1;
    }
}