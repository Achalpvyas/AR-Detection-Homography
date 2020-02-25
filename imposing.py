import cv2
import numpy as np
from imageprocessing import *

# # Imposing Lena image over the AR tag
def imposingimage(image,video):
    imgframe = cv2.imread(image)
    # cv2.imshow('Lena',imgframe)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    # print(frame.shape[1])
    # print(frame.shape[0])
    # Corners of the Lena image
    newFrames = []
    lenaCorners = np.float32([[0,0], [0,imgframe.shape[0]], [imgframe.shape[1], 0], [imgframe.shape[0], imgframe.shape[1]]])
    size = (0,0)
    # Reading the video
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        height,width,layers = frame.shape
        # Size of the each frame in the video
        size = (width,height)
        frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        # Only enters if there are corners
        pf = preprocessing(frame)
        if detectArTag(pf, frame):
            # Corners of tag and ouside 4 corners
            tagcorners = detectArTag(pf, frame)
            #if tagcorners:
            # Corners of the AR tag
            tagarray = np.array(tagcorners[1], dtype=float)
            # Homography between Lena image corners and tag image corners
            h = homography(lenaCorners, tagarray)
            # Warping Lena and AR tag
            img = warping(imgframe, h, tagarray, frame)
            # Projecting Lena onto AR tag
            #projection = projectionMatrix(h)
            # Appending frames of projected image to form new video
            newFrames.append(img)
        cv2.imshow('AR Tag',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(newFrames)
    return newFrames, size


# Funcion to be modified
def warping(frame, homography, tagcorners, image):
    temp = cv2.warpPerspective(frame, homography,(image.shape[1],image.shape[0]))
    cv2.fillConvexPoly(image,tagcorners.astype(int),0,16)
    image = image + temp
    print(image)
    return image


# Checking functionality
newFrames, size = imposingimage('./data/reference_images/Lena.png','./data/Video_dataset/Tag0.mp4')



