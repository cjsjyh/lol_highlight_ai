# ML related libraries
import torch
import numpy
import h5py
import torchvision
from torchvision import transforms

# Fundamental libraries
import time
import urllib
import cpd_auto
import math

# Media related libraries
import cv2
import librosa
from PIL import Image
from skimage import io
from matplotlib import pyplot as plt
from moviepy.editor import *

#-------------------------------------
# Extract Audio Features
#-------------------------------------
def extract_only_name(name):
    return name.replace(".mp4","").split('/')[-1]

def extract_audio_from_video(video_name):
    videoclip = VideoFileClip(video_name)
    audioclip = videoclip.audio
    if not os.path.exists("audio"):
        os.makedirs("audio")
    audioclip.write_audiofile("audio/" + extract_only_name(video_name) + ".mp3")


def extract_audio_features(video_name, per_frame):
    extract_audio_from_video(video_name)
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = frame_count = 0
    success, image = cap.read()
    while success:
        if(count != 0):
            success, image = cap.read()
        if(not success):
            break
        if(count % per_frame == 0):
            frame_count += 1
        count += 1
    cap.release()

    y, sr = librosa.load("audio/" + extract_only_name(video_name) + ".mp3")
    mfcc = librosa.feature.mfcc(y, sr, hop_length=int(sr/(fps/per_frame)) )
    mfcc = numpy.moveaxis(mfcc, 0, -1)
    # Match frame count
    mfcc = mfcc[0:frame_count,:]
    return mfcc

#-------------------------------------
# Extract Video Features
#-------------------------------------
def get_pool5(name,pool5):
    def hook(model, input, output):
        pool5[name] = output.detach()
    return hook

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
    print(torch.cuda.is_available())
    for pic_num in range(0,int(capture_ori.get(cv2.CAP_PROP_FRAME_COUNT)),full_space):
        capture_ori.set(cv2.CAP_PROP_POS_FRAMES,pic_num)
        ret, frame = capture_ori.read()
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
        if pic_num % 15 == 0:
            features.append(pool5['AdaptiveAvgPool2d'].cpu().numpy().reshape(1024))
    end = time.time()
    print(end-start)

    return numpy.array(features),numpy.array(full_features)

def get_cp(video_frames,max_cp):
    a = time.time()
    kerMat = numpy.matmul(video_frames,video_frames.transpose())
    b = time.time()
    cp_tuple = cpd_auto.cpd_auto(kerMat,max_cp,desc_rate = 1,vmax=1)
    c = time.time()
    print(b-a)
    print(c-b)
    return cp_tuple

#-------------------------------------
# Wrap data into h5 file
#-------------------------------------
def revise_to_h5(f,video_name,features=None,audio_features_20=None,gtscore=None,gtsummary=None,n_frame_per_seg=None,n_frames=None,change_point=None,picks=None,n_steps=None,user_summary=None,video_name_original=None,cp_sift=None,n_frame_per_seg_sift=None):
    print("Revising Data into h5 Format")
    if features:
        if "features" in f[video_name].keys():
            del f[video_name]["features"]
        f.create_dataset(video_name+'/features',data = features)
    if audio_features_20:
        if "audio_features_20" in f[video_name].keys():
            del f[video_name]["audio_features_20"]
        f.create_dataset(video_name+'/audio_features_20',data = audio_features_20)
    if n_frame_per_seg:
        if "n_frame_per_seg" in f[video_name].keys():
            del f[video_name]["n_frame_per_seg"]
        f.create_dataset(video_name+'/n_frame_per_seg',data=n_frame_per_seg)
    if n_frames:
        if "n_frames" in f[video_name].keys():
            del f[video_name]["n_frames"]
        f.create_dataset(video_name+'/n_frames', data = n_frames)
    if change_point:
        if "change_point" in f[video_name].keys():
            del f[video_name]["change_point"]
        f.create_dataset(video_name+'/change_point',data=change_point)
    if picks:
        if "picks" in f[video_name].keys():
            del f[video_name]["picks"]
        f.create_dataset(video_name+'/picks',data=picks)
    if n_steps:
        if "n_steps" in f[video_name].keys():
            del f[video_name]["n_steps"]
        f.create_dataset(video_name+'/n_steps',data=n_steps)
    if cp_sift:
        if "change_point_sift" in f[video_name].keys():
            del f[video_name]["change_point_sift"]
        f.create_dataset(video_name+'/change_point_sift',data=cp_sift)
    if video_name:
        if "video_name" in f[video_name].keys():
            del f[video_name]["video_name"]
        f.create_dataset(video_name+'/video_name',data=video_name)
    if n_frame_per_seg_sift:
        if "n_frame_per_seg_sift" in f[video_name].keys():
            del f[video_name]["n_frame_per_seg_sift"]
        f.create_dataset(video_name+'/n_frame_per_seg_sift',data=n_frame_per_seg_sift)


def many_video_to_h5(h5,path,games,frame_space=15,full_space=5,threshold=0.007,sift_threshold=10):
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

            with h5py.File(h5,'a') as dataset:
                if game not in dataset.keys():
                    revise_to_h5(h5,game,features=features,n_frame_per_seg=n_frame_per_seg, n_frames=n_frames,change_point=change_points,picks=picks,n_steps=n_steps,cp_sift=cp_sift,n_frame_per_seg_sift=n_frame_per_seg_sift)
                else:
                    print(f"{game} is already in h5!")
    end_making_video = time.time()
    print(f'total {end_making_video-start_making_video} seconds has taken')

def one_video_to_h5(h5,path,game,frame_space=15,full_space=5,threshold=0.007,sift_threshold=10,create= False):
    start_making_video = time.time()
    vid_name = path+game
    #----------------------------------------
    # Extract Audio Features
    #----------------------------------------
    audio_features = extract_audio_features(vid_name, frame_space)

    #----------------------------------------
    # Extract Video Features
    #----------------------------------------
    """
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
    """

    #----------------------------------------
    # Save to h5
    #----------------------------------------
    if create:
        if game not in dataset.keys():
            revise_to_h5(h5,game,audio_features_20=audio_features, features=features,n_frame_per_seg=n_frame_per_seg, n_frames=n_frames,change_point=change_points,picks=picks,n_steps=n_steps,cp_sift=cp_sift,n_frame_per_seg_sift=n_frame_per_seg_sift)
        else:
            print(f"{game} is already in h5!")
    else:
        print(game)
        revise_to_h5(h5,game,audio_features_20=audio_features)
        #revise_to_h5(h5,game,audio_features_20=audio_features,features=features,n_frame_per_seg=n_frame_per_seg,n_frames=n_frames,change_point=change_points,picks=picks,n_steps=n_steps,cp_sift=cp_sift,n_frame_per_seg_sift=n_frame_per_seg_sift)
    end_making_video = time.time()
    print(f'total {end_making_video-start_making_video} seconds has taken')

#-------------------------------------
# Main function
#-------------------------------------
if __name__ == "__main__":
    video_path = "/home/lol/total_full_video/" 
    m2 = h5py.File("/home/lol/h5_dataset/m2.h5","a")

    for game in m2.keys():
        print(game)
        one_video_to_h5(m2, video_path, game)


    
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
