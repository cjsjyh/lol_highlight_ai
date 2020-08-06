import numpy as np
import cv2
import time
import pdb
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
    img1 = img1[50:240,50:590]
    img2 = img2[50:240,50:590]
    #cv2.imshow("test",img2)
    #cv2.waitKey(0)
    # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
#    print(kp1)
    # FLANN parameters
    #if des1 is not None:
    #    print(des1.shape)
    #else :
    #    print("des1 is None")
    #if des2 is not None:
    #    print(des2.shape)
    #    print(len(kp2))
    if des2 is None:
        print("des2 is None")
        return -1,None,None,None,None,None
    #breakpoint()
    matches = flann.knnMatch(des1,des2,k=2)
    # ratio test as per Lowe's paper
    num = 0
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            num += 1
            good.append([m])
    
    return num,good,kp1,kp2,img1,img2

if __name__ == "__main__":
    capture = cv2.VideoCapture("/home/lol/tj/20200628_AF_DRX_1_highlight.mp4")
    prev_frame = np.zeros([360, 640, 3], dtype = np.uint8)
    a = time.time()
    capture.set(cv2.CAP_PROP_POS_FRAMES,400)
#    num = 4500*30
    num = 400
    while True:
        if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            capture.open("/home/lol/highlight_360.mp4")
        ret, frame = capture.read()
        if num % 15 == 0:
            sim,matches,kp1,kp2,img1,img2 = match_check(prev_frame, frame)
            print(f'{num//1800}:{num//30%60}', sim, num)
            #res=cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
            #cv2.imshow("res",res)
            #cv2.waitKey(0)

            if sim <= 10:
                print(f'{num//1800}:{num//30%60}', sim, num)
                #res=cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
                #cv2.imshow("res",res)
                #cv2.waitKey(0)
      #         cv2.destroyAllWindows()
            prev_frame = frame
        #if cv2.waitKey(1) > 0: break
        num += 1
        
    b = time.time()
    print(b-a)

    capture.release()
    cv2.destroyAllWindows()
    
