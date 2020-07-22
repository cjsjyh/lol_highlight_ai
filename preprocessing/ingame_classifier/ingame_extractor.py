#/usr/bin/env python
# coding: utf-8


root_path = './'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
print(torch.cuda.memory_allocated(device))


#------------------------
# Create Model
#------------------------
cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] # 8 + 3 =vgg11

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(46592, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# Load Model
new_net = VGG(make_layers(cfg),2,True).to(device)
new_net.load_state_dict(torch.load(root_path + "/result_model/final_model.pth"))

def is_ingame(data_set):
    with torch.no_grad():
      for num, data in enumerate(data_set):
        imgs, label = data
        imgs = imgs.to(device)
        prediction = new_net(imgs)
        return torch.argmax(prediction, 1)


import os, shutil
def clear_directory(path):
  for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    os.unlink(file_path)


import sys
import argparse
import matplotlib.pyplot as plt
from PIL import Image

import cv2
print(cv2.__version__)

import math

def FindTransitions(path_in, video_name, path_out, start = -1, until=-1, frame=60):
    print("Find Transitions Start")
    # Initialize local variables
    frame_number = 0
    save_count = 0
    last_check = 0 if start == -1 else start
    saved_frame = []
    success = True
    curInGame = False # Currently In game?
    curInGameCount = 0

    # Initialize result file
    result_file = open(path_out + f"/{video_name.replace('.mp4','')}_raw.txt", 'w+')

    # Initialize OpenCV
    print("Processing: " + path_in + video_name)
    vidcap = cv2.VideoCapture(path_in + video_name)
    print("Is Open: " + str(vidcap.isOpened()))
    success,image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("FPS: "+str(fps))

    # Initialize PyTorch transform
    trans = transforms.Compose([
      transforms.Resize((240, 426)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    while success:
        success,image = vidcap.read()
        if (not success):
            break
        # Save frame image
        if (frame_number % frame == 0 and frame_number >= frame * last_check):
            cv2.imwrite(path_out + "/temp/frame%d.jpg" % frame_number, image)     # save frame as JPEG file
            save_count += 1
            saved_frame.append(frame_number)

        # Inference per 30 saved image
        if (save_count % 30 == 0 and save_count > last_check):
            print(f"Checking Frame: {frame_number}")
            dataset = torchvision.datasets.ImageFolder(path_out, transform=trans)
            dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)
            inference_result = is_ingame(dataloader)
            # Check frame inference result
            for (index,isInGame) in enumerate(inference_result.tolist()):
                in_seconds = math.floor(saved_frame[index]/fps)
                _sec = in_seconds % 60
                _min = math.floor(in_seconds / 60) % 60
                _hr = math.floor(in_seconds / 60) / 60
                # If frame is in game
                if (isInGame == 1):
                    if (not curInGame):
                        if (curInGameCount == 2):
                            print(f"Game started {_min}:{_sec}")
                            result_file.write(f"start {saved_frame[index]} {in_seconds} {_hr}:{_min}:{_sec}\n")
                            curInGameCount = 0
                            curInGame = True
                        else:
                            if(curInGameCount < 0):
                                curInGameCount = 0
                            curInGameCount += 1
                else:
                    if (curInGame):
                        if (curInGameCount == -2):
                            print(f"Game Finished {_min}:{_sec}")
                            result_file.write(f"finish:{saved_frame[index]} {in_seconds} {_hr}:{_min}:{_sec}\n")
                            curInGameCount = 0
                            curInGame = False
                        else:
                            if(curInGameCount > 0):
                                curInGameCount = 0
                            curInGameCount += -1
            saved_frame[:] = []
            last_check = save_count
            #clear folder
            clear_directory(path_out + "/temp")

        if (until != -1 and save_count == until):
            break
        frame_number += 1
    clear_directory(path_out + "/temp")
    result_file.close()
    vidcap.release()
    print(video_name + " Done!")


# In[14]:


FindTransitions("/home/lol/lol_highlight_ai/preprocessing/downloader/full_raw/","20200228_APK_DRX_T1_SB.mp4",
"/home/lol/lol_highlight_ai/preprocessing/ingame_classifier/inference_result")


# In[ ]:


# _image = Image.open(root_path + pathOut + "/temp/frame%d.jpg" % saved_frame[index])
# plt.imshow(_image)

