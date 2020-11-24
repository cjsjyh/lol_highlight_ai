import h5py
import numpy as np
a = h5py.File("h5_dataset/a.h5","r+")
b = h5py.File("h5_dataset/b.h5","r+")
c = h5py.File("h5_dataset/c.h5","r+")
d = h5py.File("h5_dataset/d.h5","r+")
e = h5py.File("h5_dataset/e.h5","r+")
f = h5py.File("h5_dataset/f.h5","r+")

#codes for inserting Ground Truth Score to H5 format files
#there were actually 6 files before merged into one dataset files.


def whichFile(video,a,b,c,d,e,f):
    #print(a.keys())
    if video in a.keys():
        return a
    if video in b.keys():
        return b
    #print(c.keys())
    if video in c.keys():
        return c
    #print(d.keys())
    if video in d.keys():
        return d
    #print(e.keys())
    if video in e.keys():
        return e
    #print(f.keys())
    if video in f.keys():
        return f
    print("No Video Has Found")

def create_GTScore(video, h5,data):
    frame= h5[video]["n_frames"][...]
    gts = []
    user_sum = []
    idx = 0
    print(frame)
    print(data)
    for i in range(0,frame):
        if data[idx][0] > i:
            user_sum.append(0)
            if i % 15 == 0:
                gts.append(0)
        elif data[idx][0] <= i and data[idx][1] >= i:
            #print(i)
            user_sum.append(1)
            if i % 15 == 0:
                gts.append(1)
        else:
            idx += 1
            if len(data) <= idx:
                user_sum.append(0)
                if i % 15 == 0:
                    gts.append(0)
                idx -= 1
            else:
                if data[idx][0] > i:
                    user_sum.append(0)
                    if i % 15 == 0:
                        gts.append(0)
                else:
                    user_sum.append(1)
                    if i % 15 == 0:
                        gts.append(1)
    return gts,user_sum


#labeling output text file is consisted by format below
#<<number of highlighted range>> <<video name>>
#1032 1552
#1995 2666
#14023 16222
#...
#...
#function labelTextToData reads the info from the text and create labelled data


def labelTextToData(file):
    with open(file,"r") as gtFile:
        data = None
        while True:
            video = gtFile.readline()
            if video == "":
                break
            video = video.replace("\n","")
            n = int(gtFile.readline())
            data = []
            for i in range(n):
                start,end = gtFile.readline().split(' ')
                data.append((int(start),int(end)))
            h5 = whichFile(video,a,b,c,d,e,f)
            gts,us = create_GTScore(video,h5,data)
            frame =int(h5[video]["n_frames"][...])
            us = np.array(us)
            gts = np.array(gts)
            #print(h5[video]["features"][...].shape)
            #del h5[video]["gtscore"]
            if "gtscore" in h5[video].keys():
                del h5[video]["gtscore"]
            if "gtsummary" in h5[video].keys():
                del h5[video]["gtsummary"]
            if "user_summary" in h5[video].keys():
                del h5[video]["user_summary"]
            h5.create_dataset(video+"/gtscore",data=gts)
            h5.create_dataset(video+"/gtsummary",data=gts)
            h5.create_dataset(video+"/user_summary",data=us.reshape(1,-1))
        #print(h5[video]["gtscore"][...].shape)


if __name__ == "__main__":
    labelTextToData("video1_output.txt")
    labelTextToData("video2_output.txt")
#for ele in a.keys():
#    frame = a[ele]["n_steps"][...]
#    us = a[ele]["n_frames"][...]
#    fake = [0]*int(frame)
#    uss = [0]*int(us)
    #print(a[ele+'/gtscore'])
    #del a[ele+'/gtscore']
    #del a[ele+'/gtsummary']
    #dt = a[ele]["gtscore"]
    #dt[...] = np.array(fake)
    #st = a[ele]["gtsummary"]
    #st[...] = np.array(fake)
    #a[ele]["gtscore"][...] = np.array(fake)
    #a[ele]["gtsummary"][...] = np.array(fake)
    #del a[ele]["gtscore"]
    #del a[ele]["gtscummary"]
    #a.create_dataset(ele+"/gtscore",data=np.array(fake))
    #a.create_dataset(ele+"/gtsummary",data=np.array(fake))
#    del a[ele]["user_summary"]
    #print(a[ele].keys())
#    a.create_dataset(ele+'/user_summary',data=np.array(uss).reshape(1,-1))
