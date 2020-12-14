import h5py
import argparse

"""
find best scored video among evaluation videos.
command example : (python precision.py -i 0 -d data)
                    : find best video among 0th cross validation's evaluation
                    video
before run this, you need to make evaluation dataset made by main.py

"""
parser = argparse.ArgumentParser("get number of dataset")
parser.add_argument("-i", '--index', type = int, help = "index 0-5")
parser.add_argument("-d", "--data", type = str, help = "name of h5 dataset")
args = parser.parse_args()

num = str(args.index)
name = args.data
result = h5py.File("splits/"+name+'_'+num.replace("\n","")+".h5","r")
target = h5py.File(name+".h5","r")
games = result[name].keys()

max_score = 0
max_game = None

max_f1 = 0
max_f1_game = None

#둘다 0 : 0
#둘다 1 : 2
#예상값만 1 : 1
#실제값만 1 : 3

for game in games:
    gtscore = target[game]['user_summary'][...].reshape(-1)
    result_score = result[name][game]['machine_summary'][...]
    true = 0
    true_pos = 0
    #print(result['merge'][game].keys())
    length = int(target[game]['n_frames'][...])
    #print(length)
    #print(result_score.shape)
    #print(gtscore.shape)
    for i in range(result_score.shape[0]):
        if result_score[i] == 1:
            true += 1
            if gtscore[i] == 1:
                true_pos += 1
    #print(f'{game} : {true_pos/true}')
    if true_pos/true > max_score:
        max_score = true_pos/true
        max_game = game
    if result[name][game]['fm'][...] > max_f1:
        max_f1 = result[name][game]['fm'][...]
        max_f1_game = game

print(f'max : {max_game} score : {max_score}')

print(f'max : {max_f1_game} score : {max_f1}')

