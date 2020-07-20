from google.colab import drive
drive.mount('/content/gdrive')

!ls '/content/gdrive/My Drive/SKT AI Fellowship/모델/in_game_classifier/raw'

import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut, frame=60):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      if(not success):
        break
      if(count % frame == 0):
        cv2.imwrite( pathOut + "frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
    print("Extract done")

# if __name__=="__main__":
#     a = argparse.ArgumentParser()
#     a.add_argument("--pathIn", help="path to video")
#     a.add_argument("--pathOut", help="path to images")
#     args = a.parse_args()
#     print(args)
#     extractImages(args.pathIn, args.pathOut)

pathIn = '/content/gdrive/My Drive/SKT AI Fellowship/모델/in_game_classifier/raw/game1.mp4'
pathOut = '/content/gdrive/My Drive/SKT AI Fellowship/모델/in_game_classifier/train_data/'
extractImages(pathIn, pathOut, 60)
