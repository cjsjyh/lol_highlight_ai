import cv2
from skimage import io
from matplotlib import pyplot as plt
import torch
import torchvision
import urllib
from PIL import Image
from torchvision import transforms
import time
import cpd_auto
import numpy
import h5py
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

def data_to_h5(filename,video_name,features,gtscore=None,gtsummary=None,n_frame_per_seg=None,n_frames=None,change_point=None,picks=None,n_steps=None,user_summary=None,video_name_original=None):
    h5_file_name = filename
    f = h5py.File(h5_file_name,'w')
    f.create_dataset(video_name+'/features',data = features)
    #f.create_dataset(video_name+'/gtscore',data = gtscore)
    f.create_dataset(video_name+'/n_frame_per_seg',data=n_frame_per_seg)
    f.create_dataset(video_name+'/n_frames', data = n_frames)
    f.create_dataset(video_name+'/change_point',data=change_point)
    f.create_dataset(video_name+'/picks',data=picks)
    f.create_dataset(video_name+'/n_steps',data=n_steps)
    #f.create_dataset(video_name+'/gtsummary',data=gtsummary)
    #f.create_dataset(video_name+'/user_summary',data=user_summary)
    f.create_dataset(video_name+'/video_name',data=video_name)
    f.close()

if __name__ == "__main__":
    vid_name = '/home/lol/highlight_360.mp4'
    frame_space = 15
    full_space = 5
    threshold = 0.007
    
    features,full_features = get_googlenet_pool5(vid_name,frame_space,full_space)
    cp = get_cp(full_features,int(full_features.shape[0]*full_space*threshold))
    change_points = cp[0]*full_space
    picks = numpy.array([frame_space*num for num in range(0,features.shape[0])])
    n_frames = len(full_features)*full_space
    n_steps = len(features)
    change_point_end = numpy.append(change_points[1:],len(full_features)*full_space)-1 
    change_points = numpy.vstack((change_points,change_point_end)).transpose()
    n_frame_per_seg = change_points[:,1]-change_points[:,0]+1
    data_to_h5("test.h5","high360",features=features,n_frame_per_seg=n_frame_per_seg, n_frames=n_frames,change_point=change_points,picks=picks,n_steps=n_steps)


    
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
