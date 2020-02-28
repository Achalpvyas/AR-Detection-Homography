import cv2
import numpy as np
from multiple_imageprocessing import *


# Checking functionality

lenaImg = cv2.imread('./data/reference_images/Lena.png')
lenaImg = cv2.resize(lenaImg,(100,100),interpolation = cv2.INTER_AREA)
cap = cv2.VideoCapture('./data/Video_dataset/multipleTags.mp4')


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
    tagCoordinates = detectArTag(pf,frame)

    if tagCoordinates is not None:
        for i in range(len(tagCoordinates)):
            # warp lena image to Ar_tag 
            outerTagCorners = orientation(tagCoordinates[i].copy())
            H = homography(lenaCorners, outerTagCorners)
            # cv2.warpPerspective(lenaImg,H,(frame.shape[0],frame.shape[1]))
            warpFrame(lenaImg,H,(frame.shape[0],frame.shape[1]),None,frame)

            # cv2.warpPerspective(lenaImg,H,(frame.shape[0],frame.shape[1]))
            # cv2.imshow('Opencv inbuilt function',frame)

    cv2.imshow('AR Tag Using custom function',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


