import cv2

game = cv2.VideoCapture("/home/lol/total_full_video/20200205_KT_GEN_1_full.mp4")
highlight = cv2.VideoCapture("/home/lol/total_highlight_video/20200205_KT_GEN_1_highlight.mp4")
fps_highlight = highlight.get(cv2.CAP_PROP_FPS)
fps_game = game.get(cv2.CAP_PROP_FPS)

print(fps_highlight, fps_game)
