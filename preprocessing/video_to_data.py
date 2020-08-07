import cv2
from skimage import io
from matplotlib import pyplot as plt
import torch
import torchvision
import urllib
from PIL import Image
from torchvision import transforms
import time
from VASNet.cpd_auto import cpd_auto
import numpy
import h5py
from tj.highlight_cut import match_check
import os
import argparse

parser = argparse.ArgumentParser("create h5 file from lol videos")
parser.add_argument('-d', '--dataset', type=str, required=True, help="path to h5 dataset (required)")
parser.add_argument('-v', '--video', type=str, default='datasets', help="path to video")
parser.add_argument('-i','--index', type=int, default=0, help="index of process to run (max 5)")
# parser.add_argument('--num-splits', type=int, default=5, help="how many splits to generate (default: 5)")
# parser.add_argument('--train-percent', type=float, default=0.8, help="percentage of training data (default: 0.8)")

args = parser.parse_args()


def get_pool5(name,pool5):
    def hook(model, input, output):
        pool5[name] = output.detach()
    return hook
#get video objects

def get_googlenet_pool5(video_name,per_frame,full_space):
    start = time.time()
    capture_ori = cv2.VideoCapture(video_name)

    #load pretrained googlenet
    model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
    pool5 = {}
    model.avgpool.register_forward_hook(get_pool5('AdaptiveAvgPool2d',pool5))
    model.eval()

    # sample execution (requires torchvision)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = []
    full_features = []
    print(capture_ori.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = capture_ori.get(cv2.CAP_PROP_FRAME_COUNT)
    print(video_length)
    for pic_num in range(0,int(video_length)):
        ret, frame = capture_ori.read()
        if frame is None:
            if pic_num + 300 < video_length:
                continue
            print(f"{pic_num}th frame is last checked frame")
            break

        if pic_num % full_space == 0:
            capture_ori_convert = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            capture_ori_pil = Image.fromarray(capture_ori_convert)

            input_tensor = preprocess(capture_ori_pil)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                if pic_num % 10000 == 0:
                    print(pic_num)
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output = model(input_batch)
            full_features.append(pool5['AdaptiveAvgPool2d'].cpu().numpy().reshape(1024))   
            if pic_num % per_frame == 0:
                features.append(pool5['AdaptiveAvgPool2d'].cpu().numpy().reshape(1024))
    end = time.time()
    print(f"{end-start} seconds has taken from GoogleNet")

    return numpy.array(features),numpy.array(full_features)

def get_cp(video_frames,max_cp):
    a = time.time()
    kerMat = numpy.matmul(video_frames,video_frames.transpose())
    cp_tuple = cpd_auto(kerMat,max_cp,desc_rate = 1,vmax=1)
    c = time.time()
    print(f"{c-a} seconds consumed on KTS Algorithm change point dedection")
    return cp_tuple

def data_to_h5(filename,video_name,features,gtscore=None,gtsummary=None,n_frame_per_seg=None,n_frames=None,change_point=None,picks=None,n_steps=None,user_summary=None,video_name_original=None,cp_sift=None,n_frame_per_seg_sift=None):
    print("Saving Data Into h5 Format ... ")
    h5_file_name = filename
    f = h5py.File(h5_file_name,'r+')
    f.create_dataset(video_name+'/features',data = features)
    #f.create_dataset(video_name+'/gtscore',data = gtscore)
    f.create_dataset(video_name+'/n_frame_per_seg',data=n_frame_per_seg)
    f.create_dataset(video_name+'/n_frames', data = n_frames)
    f.create_dataset(video_name+'/change_point',data=change_point)
    f.create_dataset(video_name+'/picks',data=picks)
    f.create_dataset(video_name+'/n_steps',data=n_steps)
    #f.create_dataset(video_name+'/gtsummary',data=gtsummary)
    #f.create_dataset(video_name+'/user_summary',data=user_summary)
    f.create_dataset(video_name+'/change_point_sift',data=cp_sift)
    f.create_dataset(video_name+'/video_name',data=video_name)
    f.create_dataset(video_name+'/n_frame_per_seg_sift',data=n_frame_per_seg_sift)
    f.close()

def change_point_sift(video_path,start_frame,full_space,threshold):
    print("Making Change Point Using SIFT...")
    capture = cv2.VideoCapture(video_path)
    prev_frame = numpy.zeros([360, 640, 3], dtype = numpy.uint8)
    num = start_frame
    cp_list = []
    frame_length = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    start = time.time()
    while num < frame_length:
        ret, frame = capture.read()
        if frame is None:
            if num + 300 < frame_length:
                continue
            print(f"{num}th frame is last checked frame")
            break
        if num % full_space == 0:
            sim,matches,kp1,kp2,img1,img2 = match_check(prev_frame, frame)
            if sim == -1:
                pass
            elif sim <= threshold:
                cp_list.append(num)
            prev_frame = frame
        num += 1
    capture.release()
    end = time.time()
    print(f"{end-start} seconds consumed on SIFT Change Point Detection")
    return numpy.array(cp_list)

def one_video_to_h5(h5,path,games,frame_space=15,full_space=5,threshold=0.007,sift_threshold=10):
    if torch.cuda.is_available():
        print("GPU is available")

    start_making_video = time.time()
    for game in games:
        if '.mp4' in game:
            vid_name = path+game
            print(vid_name)
            features,full_features = get_googlenet_pool5(vid_name,frame_space,full_space)
            cp = get_cp(full_features,int(full_features.shape[0]*full_space*threshold))
            picks = numpy.array([frame_space*num for num in range(0,features.shape[0])])
            n_frames = len(full_features)*full_space
            n_steps = len(features)

            #change point detection using KTS
            change_points = cp[0]*full_space
            change_point_end = numpy.append(change_points[1:],len(full_features)*full_space)-1 
            change_points = numpy.vstack((change_points,change_point_end)).transpose()
            n_frame_per_seg = change_points[:,1]-change_points[:,0]+1
            #change point detection using SIFT
            cp_sift = change_point_sift(vid_name,0,full_space,sift_threshold)
            cp_sift_end = numpy.append(cp_sift[1:],len(full_features)*full_space)-1 
            cp_sift = numpy.vstack((cp_sift,cp_sift_end)).transpose()    
            n_frame_per_seg_sift = cp_sift[:,1]-cp_sift[:,0]+1

            with h5py.File(h5,'r+') as dataset:
                if game not in dataset.keys():
                    data_to_h5(h5,game,features=features,n_frame_per_seg=n_frame_per_seg, n_frames=n_frames,change_point=change_points,picks=picks,n_steps=n_steps,cp_sift=cp_sift,n_frame_per_seg_sift=n_frame_per_seg_sift)
    end_making_video = time.time()
    print(f'total {end_making_video-start_making_video} seconds has taken')
    
if __name__ == "__main__":
    frame_space = 15
    full_space = 5
    threshold = 0.007
    sift_threshold = 10
    game_count = 0
    path = args.video
    h5_path = '/home/lol/test.h5'
    games = [i for i in sorted(os.listdir(args.video))]
    game_total = len(games)
    game_per_seg = int(game_total/6)
    if args.index == 5:
        games = games[args.index*game_per_seg:]
    else:
        games = games[args.index*game_per_seg:(args.index+1)*game_per_seg]
    one_video_to_h5(args.dataset,path,games,frame_space,full_space,threshold,sift_threshold)

    

    
    #data_to_h5(filename='ex1.h5',video_name=vid_name,features = a,gtscore )
#***********************************************************************************************************************************************
#/key
#    /features                 2D-array with shape (n_steps, feature-dimension)
#    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
#    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
#    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
#    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
#    /n_frames                 number of frames in original video
#    /picks                    posotions of subsampled frames in original video
#    /n_steps                  number of subsampled frames
#    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
#    /video_name (optional)    original video name, only available for SumMe dataset
#***********************************************************************************************************************************************
