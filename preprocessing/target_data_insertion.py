import h5py
import numpy as np
a = h5py.File("a.h5","r+")
b = h5py.File("b.h5","r+")
c = h5py.File("c.h5","r+")
d = h5py.File("d.h5","r+")
e = h5py.File("e.h5","r+")
f = h5py.File("f.h5","r+")

def whichFile(video,a,b,c,d,e,f):
    if video in a.keys():
        return a
    if video in b.keys():
        return b
    if video in c.keys():
        return c
    if video in d.keys():
        return d
    if video in e.keys():
        return e
    if video in f.keys():
        return f

def create_GTScore(video, h5,data):
    frame= h5[video]["n_frames"][...]
    gts = []
    idx = 0
    for i in range(0,frame,15):
        if data[idx][0] > i:
            gts.append(0)
        if data[idx][0] <= i and data[idx][1] >= i:
            gts.append(1)
        else:
            idx += 1
            if data[idx][0] > i:
                gts.append(0)
    return gts



with open("gtscore.txt","r") as gtFile:
    data = None
    while True:
        video = gtFile.readline()
        n = int(gtFile.readline())
        data = []
        for i in range(n):
            data.append((gtFile.readline().split(" ")))
        h5 = whichFile(video,a,b,c,d,e,f)
        gts = create_GTScore(video,h5,data)
        print(gts)
        #h5.create_dataset(video+"/gtscore",dataset=np.array(gts))

