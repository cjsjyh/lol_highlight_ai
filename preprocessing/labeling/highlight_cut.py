import numpy as np
import cv2

sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

def match_check(img1, img2):
    if type(img1) != np.ndarray:
        img1 = np.array(img1)
    if type(img2) != np.ndarray:
        img2 = np.array(img2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    
    matches = flann.knnMatch(des1,des2,k=2)
    # ratio test as per Lowe's paper
    num = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            num += 1
    
    return num

if __name__ == "__main__":
    capture = cv2.VideoCapture("/home/lol/highlight_360.mp4")
    prev_frame = np.zeros([360, 640, 3], dtype = np.uint8)

    num = 1
    while True:
        if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            capture.open("/home/lol/highlight_360.mp4")


        ret, frame = capture.read()
        if num % 30 == 0:
            sim = match_check(prev_frame, frame)
            if sim <= 300:
                print(f'{num//1800}:{num//30%60}', sim)
            prev_frame = frame

        if cv2.waitKey(1) > 0: break
        num += 1

    capture.release()
    cv2.destroyAllWindows()
