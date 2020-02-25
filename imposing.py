import cv2
import numpy as np
from detection import homography

# # Imposing Lena image over the AR tag
def imposingimage(image,video):
    imgframe = cv2.imread(image)
    cv2.imshow('Lena',imgframe)
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

def detectArTag(pf,frame):
    contours,hierarchy = cv2.findContours(pf,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    l = []
    corners = []
    detectedCorners = np.zeros((2,1))
    for i in range(len(hierarchy[0])):

        #filter1->look for contours with parent
        #       ->Tag must be in a detectable background
       if(hierarchy[0][i][3] != -1):

            #filter2->look for contours with more than 4 corners(inner portion)
            perimeter1 = cv2.arcLength(contours[i], True)
            approx1 = cv2.approxPolyDP(contours[i], 0.02 * perimeter1, True)
            if(len(approx1)>4):

                #filter3-> look for contours with quadilateral parents
                parentId = hierarchy[0][i][3]
                perimeter2 = cv2.arcLength(contours[parentId],True)
                approx2 =cv2.approxPolyDP(contours[parentId],0.02*perimeter2,True)
                if(len(approx2)==4):
                        l.append(i)
                        l.append(parentId)
                        corners.append(approx1)
                        corners.append(approx2)

                    # To be modified
                    #filter4 -> thresholding contours with areas
                    # areaOfChild = cv2.contourArea(contours[i])
                    # areaOfParent = cv2.contourArea(contours[parentId])
                    # diffarea = areaOfParent-areaOfChild
                    # print(areaOfChild)
                    # print(areaOfParent)
                    # print(areaOfParent-areaOfChild)
                    # print("----------")
                    # if(abs(diffarea)>100 and abs(diffarea)<1000):
                        # l.append(i)
                        # l.append(parentId)

    # hull= [cv2.convexHull(contours[i],False) for i in l]
    # cv2.drawContours(frame,hull,-1,(0,255,0),8)
    filteredContours = [contours[l[1]] for i in l]
    cv2.drawContours(frame,filteredContours,-1,(0,0,255),3)
   
    if(len(corners)!=2):
        return None
   
    outertag = np.float32(corners[1])
    outertag = outertag[:,0,:]
    idx=np.argsort(np.sum(outertag,axis = 1))
    outertag = outertag[idx]

    innertag = np.float32(corners[0])
    innertag = innertag[:,0,:]
    
    return [innertag,outertag]

def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    edges = cv2.Canny(blur,100,200)
    return edges 

# Checking functionality
newFrames, size = imposingimage('Lena.png','Tag0.mp4')
video(newFrames, size)
