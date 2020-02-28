from imageprocessing import *
import numpy as np
import cv2
import matplotlib.pyplot as plt


######################################################
#               Process Frame
######################################################
def processFrame(frame):
    pf = preprocessing(frame)
    tagCoordinates = detectArTag(pf,frame)
    if(tagCoordinates is not None):
        desiredCoordinates =  np.float32([[0,0],[200,0],[0,200],[200,200]])

        hmat = homography(tagCoordinates[1],desiredCoordinates)
        warpedtag = cv2.warpPerspective(frame,hmat,(200,200))
        tagId = retrieveInfo(warpedtag,1) 
        tagstr = "tag detected - " + ''.join(str(e) for e in tagId)
        frame2 = frame.copy()
        cv2.putText(frame2,tagstr,(280,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)
        cv2.imshow('AR Tag using inbuilt Opencv wrap function',frame2)
        
        warpedtag = warpFrame(frame,hmat,(200,200),tagCoordinates[1])
        tagId = retrieveInfo(warpedtag,0) 
        tagstr = "tag detected - " + ''.join(str(e) for e in tagId)
        cv2.putText(frame,tagstr,(280,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)
        cv2.imshow('AR Tag using custom wrap function',frame2)


######################################################
#              Reading Video 
#####################################################
cap = cv2.VideoCapture('./data/Video_dataset/Tag1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    processFrame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





# dst = cv2.cornerHarris(fp,2,3,0.04)
# dst = cv2.dilate(dst,None,iterations=0)
# frame[dst>0.01 * dst.max()] =[0,255,0]

# dst = cv2.cornerHarris(gray,2,3,0.04)
# ratio = img.shape[0]/300
# img = cv2.imread('./data/reference_images/ref_marker.png')
# detectArTag(img)

# gray = np.float32(gray)
# canny = cv2.Canny(img,100,200)


# erode =cv2.erode(thresh,None, iterations=3)
# dilate = cv2.dilate(thresh,None,iterations=3)





