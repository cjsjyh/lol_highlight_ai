
#IMPORTS
import cv2
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
# from torchsummary import summary
import requests
from PIL import Image
import h5py
game = h5py.File('tj/h5_dataset/a.h5', 'r')

#Using VGG-19 pretrained model for image classification
model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)

# model = torchvision.models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

def download(url,fname):
    response = requests.get(url)
    with open(fname,"wb") as f:
        f.write(response.content)
    
# Downloading the image    
# download("https://specials-images.forbesimg.com/imageserve/5db4c7b464b49a0007e9dfac/960x0.jpg?fit=scale","input.jpg")

# Opening the image
# img = Image.open('input.jpg') 


# Preprocess the image
def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

'''
    Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
    => Y/(1/σ) follows Distribution(0,σ)
    => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
'''
def deprocess(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(image)

def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))


capture_ori = cv2.VideoCapture('/home/lol/highlight_360.mp4')
capture_ori.set(cv2.CAP_PROP_POS_FRAMES,25010)
for pic_num in range(3500,int(capture_ori.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = capture_ori.read()
        capture_ori_convert = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        capture_ori_pil = Image.fromarray(capture_ori_convert)

        input_tensor = preprocess(capture_ori_pil)
        print('tensor :', input_tensor.shape)

        input_batch = input_tensor.unsqueeze(0)
        print(input_batch.shape)

        # cv2.imshow('sex', input_tensor)

        break

# preprocess the image
# X = preprocess(input_batch[0])
# we would run the model in evaluation mode
model.eval()

# we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
input_batch.requires_grad_()

'''
forward pass through the model to get the scores, note that VGG-19 model doesn't perform softmax at the end
and we also don't need softmax, we need scores, so that's perfect for us.
'''

scores = model(input_batch)

# Get the index corresponding to the maximum score and the maximum score itself.
score_max_index = scores.argmax()
score_max = scores[0,score_max_index]

'''
backward function on score_max performs the backward pass in the computation graph and calculates the gradient of 
score_max with respect to nodes in the computation graph
'''
score_max.backward()

'''
Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
across all colour channels.
'''
saliency, _ = torch.max(input_batch.grad.data.abs(),dim=1)

# code to plot the saliency map as a heatmap
fig=plt.figure(figsize=(100, 200))
fig.add_subplot(1, 3, 1)
plt.imshow(input_tensor[0])
fig.add_subplot(1, 3, 2)
img = Image.open('s.jpg') 
# img = T.functional.crop(img, 50, 50, 290, 770)
newnew =T.Compose([
      T.Resize((240, 426)),
      T.ToTensor(),
    ])
frame = newnew(img)
print('asdfadfadfasfd', frame.shape)
frame=frame[:, 25:145, 25:385]
print('aewf', frame.shape)
print(type(frame))
torchvision.utils.save_image(frame, "/home/lol/Desktop/a.jpg")
# farme.save_image
exit()
cv2.imshow('a', frame)
fig.add_subplot(1, 3, 3)

plt.imshow(saliency[0], cmap=plt.cm.hot)
# plt.axis('off')
plt.show()