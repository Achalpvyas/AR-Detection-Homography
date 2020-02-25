import cv2
import numpy as np
from imageprocessing import *


# Checking functionality

lenaImg = cv2.imread('./data/reference_images/Lena.png')
lenaImg = cv2.resize(lenaImg,(100,100),interpolation = cv2.INTER_AREA)
cap = cv2.VideoCapture('./data/Video_dataset/Tag0.mp4')


while(cap.isOpened()):
    # Reading the video
    ret, frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    print(lenaImg.dtype)
    
    #obtaining lena corners to warp image
    x,y,_ = lenaImg.shape
    lenaCorners = np.float32([[0,0], [0,x],[y,0],[x,y]])

    # Only enters if there are corners
    pf = preprocessing(frame)
    if detectArTag(pf, frame):

        # warp lena image to Ar_tag 
        outerTagCorners = (detectArTag(pf, frame)[1]).astype('float32')
        H = homography(lenaCorners, outerTagCorners)
        warpFrame(lenaImg,H,(frame.shape[0],frame.shape[1]),None,frame)

    cv2.imshow('AR Tag',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
        
cap.release()
cv2.destroyAllWindows()


