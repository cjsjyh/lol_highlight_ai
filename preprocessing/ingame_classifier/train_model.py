#!/usr/bin/env python
# coding: utf-8

#from google.colab import drive
#drive.mount('/content/gdrive')

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

#------------------------
# Load data
#------------------------
trans = transforms.Compose([
  transforms.Resize((240, 426)),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root_path = '/content/gdrive/My Drive/SKT AI Fellowship/모델/in_game_classifier'
train_data = torchvision.datasets.ImageFolder(root=root_path + '/train_data', transform=trans)
test_data = torchvision.datasets.ImageFolder(root=root_path + '/test_data', transform=trans)

train_set = DataLoader(dataset = train_data, batch_size=15, shuffle = True, num_workers=2)
test_set = DataLoader(dataset = test_data, batch_size=len(test_data))

cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] # 8 + 3 =vgg11

#------------------------
# Network
#------------------------
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


#------------------------
# Initialization
#------------------------
net = VGG(make_layers(cfg),2,True).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss().to(device)


#------------------------
# Training
#------------------------
total_batch = len(train_set)
epochs = 5
for epoch in range(epochs):
  avg_cost = 0.0
  for num, data in enumerate(train_set):
    imgs, labels = data
    imgs = imgs.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    hypothesis = net(imgs)
    loss = loss_func(hypothesis, labels)
    loss.backward()
    optimizer.step()

    avg_cost += loss / total_batch
  print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))
  torch.save(net.state_dict(), root_path + "/result_model/epoch{}_model.pth".format(epoch+1))
print('Learning Finished!')
torch.save(net.state_dict(), root_path + "/result_model/final_model.pth")

#------------------------
# Evaluation
#------------------------
new_net = VGG(make_layers(cfg),2,True).to(device)
new_net.load_state_dict(torch.load(root_path + "/result_model/final_model.pth"))

with torch.no_grad():
  for num, data in enumerate(test_set):
    imgs, label = data
    imgs = imgs.to(device)
    label = label.to(device)

    prediction = new_net(imgs)

    correct_prediction = torch.argmax(prediction, 1) == label
    print(torch.argmax(prediction, 1))
    print(torch.argmax(prediction, 1).tolist())

    accuracy = correct_prediction.float().mean()
    print("Accuracy: ", accuracy.item())

