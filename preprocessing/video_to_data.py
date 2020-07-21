import cv2
from skimage import io
from matplotlib import pyplot as plt
import torch
import torchvision
import urllib
from PIL import Image
from torchvision import transforms


def get_pool5(name,pool5):
    def hook(model, input, output):
        pool5[name] = output.detach()
    return hook
#get video objects
def get_googlenet_pool5(video_name,per_frame):
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
        
    for pic_num in range(0,int(capture_ori.get(cv2.CAP_PROP_FRAME_COUNT)),per_frame):
        capture_ori.set(cv2.CAP_PROP_POS_FRAMES,pic_num)
        ret, frame = capture_ori.read()
        capture_ori_convert = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        capture_ori_pil = Image.fromarray(capture_ori_convert)

        input_tensor = preprocess(capture_ori_pil)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        features.append(pool5['AdaptiveAvgPool2d'])    
    return features

a = get_googlenet_pool5('20200422_T1_DRX.mp4',15)
print(len(a))
