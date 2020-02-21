import numpy as np
import cv2
import matplotlib.pyplot as plt

##########################################
#       
##########################################
def detectArTag(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    dst = cv2.dilate(dst,None)
    img[dst>0.01 * dst.max()] =[0,0,255]






img = cv2.imread('./data/reference_images/ref_marker.png')
detectArTag(img)







# cv2.imshow("Harris Corner Detection",img)

# canny = cv2.Canny(img,100,200)
# cv2.imshow("Canny", canny)

# if(cv2.waitKey(0) & 0xff == 27):
    # cv2.destroyAllwindows()














# cap = cv2.VideoCapture('vtest.avi')

# while(cap.isOpened()):
    # ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('frame',gray)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

# cap.release()
# cv2.destroyAllWindows()

