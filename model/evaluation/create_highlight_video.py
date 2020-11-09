import h5py
import numpy as np
from moviepy.editor import VideoFileClip as vc, concatenate_videoclips as con

num = input("what number").replace("\n","")
h5 = h5py.File("merge_"+num+".h5","r")
game = h5['merge'].keys()
#print(h5['merge']['20200208_GEN_GRF_2_full.mp4'].keys())

#print(game)
#game = game.replace('\n','')

#clip1 = vc("/home/lol/total_full_video/"


for each in game:
    start = 0
    end = 0
    concat = []
    clip= vc("/home/lol/total_full_video/"+each)
    print(h5['merge'][each].keys())
    print(h5['merge'][each]['score'][...])
    print(h5['merge'][each]['att'][...])
    print(h5['merge'][each]['fm'][...])
    for idx,val in enumerate(h5['merge'][each]['machine_summary'][...]):
        if val != 0:
            if start == 0:
                start = idx
            else:
                end = idx
        else:
            if start != 0:
                if start-end > 15:
                    cut = clip.subclip(start/30,end/30)
                    concat.append(cut)
            start = 0
    summary = con(concat)
    summary.write_videofile('sum_videos/'+num+'/'+each.replace("full","sum"))
