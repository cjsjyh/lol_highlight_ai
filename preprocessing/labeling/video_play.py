import numpy as np
import cv2
import re

game = cv2.VideoCapture("20200205_KT_GEN_1_full.mp4")
highlight = cv2.VideoCapture("20200205_KT_GEN_1_highlight.mp4")
fps_highlight = highlight.get(cv2.CAP_PROP_FPS)
fps_game = game.get(cv2.CAP_PROP_FPS)
print(game.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
highlight.set(cv2.CAP_PROP_POS_FRAMES, 415)
game.set(cv2.CAP_PROP_POS_FRAMES, 4465)

_, h_frame = highlight.read()
_, f_frame = game.read()
cv2.imshow('highlight', h_frame)
cv2.imshow('game', f_frame)
numre = re.compile('[0-9]')
print(f'highlight frame : {highlight.get(cv2.CAP_PROP_POS_FRAMES)} / game frame : {game.get(cv2.CAP_PROP_POS_FRAMES)}')
print('press a key to read next frame of full game or s key for highlight and press d key to read both frame.')
while (True):
    keyval = cv2.waitKey(0)
    if keyval == 97: # input 'a' to read next frame of full game.
        _, f_frame = game.read()
        print(f'game frame : {game.get(cv2.CAP_PROP_POS_FRAMES)}')
        cv2.imshow('game', f_frame)
    elif keyval == 115: # input 's' to read next frame of highlight.
        _, h_frame = highlight.read()
        print(f'highlight frame : {highlight.get(cv2.CAP_PROP_POS_FRAMES)}')
        cv2.imshow('highlight', h_frame)
    elif keyval == 100: # input 'd' to read next frame of highlight.
        _, h_frame = highlight.read()
        print(f'highlight frame : {highlight.get(cv2.CAP_PROP_POS_FRAMES)}')
        cv2.imshow('highlight', h_frame)
        _, f_frame = game.read()
        print(f'game frame : {game.get(cv2.CAP_PROP_POS_FRAMES)}')
        cv2.imshow('game', f_frame)
    elif keyval == 113: #input 'q' to read specified fraem of game.
        try:
            num = int(''.join(numre.findall(input('Input order of game frame you want to read : '))))
        except ValueError:
            print('Input only integer!')
            continue
        print(f'{num} / {type(num)}')
        game.set(cv2.CAP_PROP_POS_FRAMES, num)
        _, f_frame = game.read()
        print(f'game frame : {game.get(cv2.CAP_PROP_POS_FRAMES)}')
        cv2.imshow('game', f_frame)
    elif keyval == 119: #input 'w' to read specified fraem of highlight.
        try:
            num = int(''.join(numre.findall(input('Input order of highlight frame you want to read : '))))
        except ValueError:
            print('Input only integer!')
            continue
        highlight.set(cv2.CAP_PROP_POS_FRAMES, num)
        _, h_frame = highlight.read()
        print(f'highlight frame : {highlight.get(cv2.CAP_PROP_POS_FRAMES)}')
        cv2.imshow('highlight', h_frame)
    else:
        break

#capture.release()
cv2.destroyAllWindows()
