import cv2
import numpy as np
from imageprocessing import *


# # Imposing Lena image over the AR tag
def imposingimage(image,video):
    lenaImg = cv2.imread(image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    
    #obtaining lena corners to warp image
    x,y,_ = lenaImg.shape
    lenaCorners = np.float32([[0,0], [0,x],[y,0],[x,y]])

    # Reading the video
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        height,width,layers = frame.shape

        frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        # Only enters if there are corners
        pf = preprocessing(frame)
        if detectArTag(pf, frame):

            # warp lena image to Ar_tag 
            outerTagCorners = (detectArTag(pf, frame)[1]).astype('float')
            H = homography(lenaCorners, outerTagCorners)
            img = warping(lenaImg, H, outerTagCorners, frame)

        cv2.imshow('AR Tag',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Funcion to be modified
def warping(image, homography, tagcorners, frame):
    temp = cv2.warpPerspective(image, homography,(frame.shape[1],frame.shape[0]))
    cv2.fillConvexPoly(frame,tagcorners.astype(int),0,1)

    # temp = warpFrame(image, homography,(frame.shape[0],frame.shape[1]),tagcorners)
    frame = frame + temp
    return frame


# Checking functionality
newFrames, size = imposingimage('./data/reference_images/Lena.png','./data/Video_dataset/Tag0.mp4')



