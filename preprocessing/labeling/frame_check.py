import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time

highlight = cv2.VideoCapture("/home/sinunu/Desktop/highlight_360.mp4")
game = cv2.VideoCapture("/home/sinunu/Desktop/game_360.mp4")

fps_highlight = highlight.get(cv2.CAP_PROP_FPS)
fps_game = game.get(cv2.CAP_PROP_FPS)
h_num = 0
f_num = 0

highlight.set(cv2.CAP_PROP_POS_FRAMES, 20 * fps_highlight)
game.set(cv2.CAP_PROP_POS_FRAMES, 4200 * fps_game)

f_num = int(4200 * fps_game)
# while h_num < 20 * fps_highlight:
#     _, h_frame = highlight.read()
#     h_num += 1

# while f_num < fps_game * 4200:
#     _, f_frame = game.read()
#     f_num += 1


print('start')
while True:
    _, h_frame = highlight.read()
    cv2.imshow('aaa', h_frame)

    obj = cv2.quality.QualityGMSD_create(h_frame)
    start = time.time()
    while f_num <= 5400 * fps_game:
        _, f_frame = game.read()
        
        # if f_num % 30 == 0:
        #     print(f'f_num : {f_num/fps_game//60}:{f_num/fps_game%60}')
        sim = obj.compute(f_frame)
        # print(f'f_num : {int(f_num/fps_game//60)}:{int(f_num/fps_game)%60} => {sim}')
        if sum(sim) < 0.3:
            print('find!!!!', sim)
            h_num += 1
        # cv2.imshow('frame', f_frame)
        if cv2.waitKey(1) == 27:
            exit()
            
        f_num += 1
    print(f'time : {time.time() - start} seconds')
    exit()
        
    
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()