import h5py
import numpy as np
from moviepy.editor import VideoFileClip as vc, concatenate_videoclips as con
import argparse

parser = argparse.ArgumentParser("get number of video")
parser.add_argument('-i', '--index', type=int, help="index 0-5")
parser.add_argument('-d', '--data', type=str, help="name of h5 dataset")
args = parser.parse_args()

num = str(args.index)
name = str(args.data)
h5 = h5py.File(name+"_"+num+".h5","r")
game = h5[name].keys()
#print(h5['merge']['20200208_GEN_GRF_2_full.mp4'].keys())

#print(game)
#game = game.replace('\n','')

#clip1 = vc("/home/lol/total_full_video/"


for each in game:
    print(each)
    start = 0
    end = 0
    concat = []
    clip= vc("/home/lol/total_full_video/"+each)
    print(h5[name][each].keys())
    print(h5[name][each]['score'][...])
    print(h5[name][each]['att'][...])
    print(h5[name][each]['fm'][...])
    for idx,val in enumerate(h5[name][each]['machine_summary'][...]):
        if val != 0:
            if start == 0:
                start = idx
            else:
                end = idx
        else:
            if start != 0:
                if end/30 > clip.duration:
                    print(end,clip.duration)
                    end = clip.duration
                    cut = clip.subclip(start/30,end)
                    concat.append(cut)
                    break
                if start/30 > clip.duration:
                    break
                print(start,end)
                cut = clip.subclip(int(start/30),int(end/30))
                concat.append(cut)
            start = 0
    summary = con(concat)
    summary.write_videofile('sum_videos/'+num+'/'+each.replace("full","sum"))