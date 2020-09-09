import h5py
import argparse

parser = argparse.ArgumentParser("get number of dataset")
parser.add_argument("-i", '--index', type = int, help = "index 0-5")
parser.add_argument("-d", "--data", type = str, help = "name of h5 dataset")
args = parser.parse_args()

num = str(args.index)
name = args.data
result = h5py.File(name+'_'+num.replace("\n","")+".h5","r")
target = h5py.File(name+".h5","r")
games = result[name].keys()

max_score = 0
max_game = None

max_f1 = 0
max_f1_game = None
"""
pred = result['merge']['20200208_APK_DWG_1_full.mp4']['machine_summary'][...]
truth = target['20200208_APK_DWG_1_full.mp4']['user_summary'][...].reshape(-1)

ans = []
for ele in range(pred.shape[0]):
    if pred[ele] > 0:
        if truth[ele] >0:
            ans.append(2)
        else:
            ans.append(1)
    elif truth[ele] > 0:
        ans.append(3)
    else:
        ans.append(0)
file = open("apkdwg.txt","w")
for ele in ans:
    file.write(str(ele))
    """
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

