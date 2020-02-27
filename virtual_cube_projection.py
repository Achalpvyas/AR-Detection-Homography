import cv2
import numpy as np
from imageprocessing import *


# Function for determining the 2-D coordinates of the cube
def cubeImpose(projectionMatrix,frame):
    # print(projectionMatrix.shape)
    # print('----------------------------------')
    cubeCoordinates = np.float32([[0,0,0,1],[0,512,0,1],[512,512,0,1],[512,0,0,1],[0,0,-512,1],[0,512,-512,1],[512,512,-512,1],[512,0,-512,1]])
    Proj= np.matmul(cubeCoordinates,projectionMatrix.T)
    # Converting the projection points to homogenous coordinates
    x,y = Proj.shape
    # Homogenous co-ordinates considered only in 2-D frame
    homogenousProj = np.zeros((x,y-1))
    for i in range(0,x):
        for j in range(0,y):
            # Not considering the z - coordinnate
            if j != y-1:
                homogenousProj[i,j] = Proj[i,j] / Proj[i,y-1]
    # Verifying matrix
    # print('----------------------------------')
    # print('homogenous matrix')
    # print([np.asarray([homogenousProj[0],homogenousProj[1]])])
    # print('-----------------------------------')
    cubeSurfaceDrawing(frame,homogenousProj)
    return frame


# Function to draw the sufaces of the cube
def cubeSurfaceDrawing(frame, cubecoordinates):
    cubecoordinates = np.int32(cubecoordinates)
    # Verifying points
    # print([np.asarray([cubecoordinates[0],cubecoordinates[1]])])
    # Inserting a cube in green colour
    # Bottom surface
    frame = cv2.drawContours(frame, [cubecoordinates[:4]],-1,(0,255,0),-3)
    # Top surface
    frame = cv2.drawContours(frame, [cubecoordinates[4:8]],-1,(0,255,0),-3)
    # Four sides
    frame = cv2.drawContours(frame, [np.asarray([cubecoordinates[0],cubecoordinates[4],cubecoordinates[5],cubecoordinates[1]])],-1,(0,255,0),-3)
    frame = cv2.drawContours(frame, [np.asarray([cubecoordinates[1],cubecoordinates[5],cubecoordinates[6],cubecoordinates[2]])],-1,(0,255,0),-3)
    frame = cv2.drawContours(frame, [np.asarray([cubecoordinates[2],cubecoordinates[6],cubecoordinates[7],cubecoordinates[3]])],-1,(0,255,0),-3)
    frame = cv2.drawContours(frame, [np.asarray([cubecoordinates[3],cubecoordinates[7],cubecoordinates[4],cubecoordinates[0]])],-1,(0,255,0),-3)
    
    return frame

# Function to vitually project the cube
def virtualCubeProjection(image,video):
    lenaImg = cv2.imread(image)
    lenaImg = cv2.resize(lenaImg,(100,100),interpolation = cv2.INTER_AREA)
    cap = cv2.VideoCapture(video)
    previousCorners = 0
    firstiteration = 0
    while(cap.isOpened()):
        # Reading the video
        ret, frame = cap.read()
        frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        print(lenaImg.shape)
        
        #obtaining lena corners to warp image
        # x,y,_ = lenaImg.shape
        x,y = 511,511
        cubeCorners = np.float32([[0,0], [0,x],[y,0],[x,y]])
        if firstiteration == 0:
            previousCorners = 0
        else:
            previousCorners = outerTagCorners
        # Only enters if there are corners
        pf = preprocessing(frame)
        if detectArTag(pf, frame):
            # warp lena image to Ar_tag 
            outerTagCorners = (detectArTag(pf, frame)[1]).astype('float32')
            H = homography(cubeCorners, outerTagCorners)
            warpFrame(lenaImg,H,(frame.shape[0],frame.shape[1]),None,frame)
            proj = projectionMatrix(H)
            image = cubeImpose(proj, frame)
        else:
            outerTagCorners = previousCorners
        cv2.imshow('Cube superimposition',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function call
virtualCubeProjection('./data/reference_images/Lena.png', './data/Video_dataset/Tag0.mp4')







