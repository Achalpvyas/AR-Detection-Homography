import numpy as np
import cv2
import matplotlib.pyplot as plt


def detectArTag(pf,frame):
    contours,hierarchy = cv2.findContours(pf,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    l = []
    corners = []
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
    filteredContours = [contours[i] for i in l]
    cv2.drawContours(frame,filteredContours,-1,(0,0,255),3)
    return corners 


def retrieveInfo():
    pass


# To compute homography between world and camera coordinates
def homography(world_coordinates, pixel_coodinates):
    xw1 = world_coordinates[0]
    xw2 = world_coordinates[1]
    xw3 = world_coordinates[2]
    xw4 = world_coordinates[3]
    yw1 = world_coordinates[4]
    yw2 = world_coordinates[5]
    yw3 = world_coordinates[6]
    yw4 = world_coordinates[7]

    xc1 = pixel_coodinates[0]
    xc2 = pixel_coodinates[1]
    xc3 = pixel_coodinates[2]
    xc4 = pixel_coodinates[3]
    yc1 = pixel_coodinates[4]
    yc2 = pixel_coodinates[5]
    yc3 = pixel_coodinates[6]
    yc4 = pixel_coodinates[7]

    A = np.array([[-xw1, -yw1, -1, 0, 0, 0, xw1 * xc1, yw1 * xc1, xc1],
              [0, 0, 0, -xw1, -yw1, -1, xw1 * yc1, yw1 * yc1, yc1],
              [-xw2, -yw2, -1, 0, 0, 0, xw2 * xc2, yw2 * xc2, xc2],
              [0, 0, 0, -xw2, -yw2, -1, xw2 * yc2, yw2 * yc2, yc2],
              [-xw3, -yw3, -1, 0, 0, 0, xw3 * xc3, yw3 * xc3, xc3],
              [0, 0, 0, -xw3, -yw3, -1, xw3 * yc3, yw3 * yc3, yc3],
              [-xw4, -yw4, -1, 0, 0, 0, xw4 * xc4, yw4 * xc4, xc4],
              [0, 0, 0, -xw4, -yw4, -1, xw4 * yc4, yw4 * yc4, yc4], ])

    [u, sigma, v] = svd(A)

    homography_matrix = v[:,8]/v[8,8]
    homography_matrix = np.reshape((3,3))

    return homography_matrix

# For camera pose estimation
def projectionMatrix(homographyMatrix):
    intrinsicParameters =np.array([1406.08415449821,0,0],
                                  [2.20679787308599, 1417.99930662800,0],
                                  [1014.13643417416, 566.347754321696,1])

    intrinsicParameters = np.transpose()

    B = np.matmul(np.inv(intrinsicParameters), homographyMatrix)
    if np.linalg.det(B) < 0:
        B = -1*B
    
    magnitude1 = np.linalg.norm(np.matmul(np.inv(intrinsicParameters),homographyMatrix[:,0]))
    magnitude2 = np.linalg.norm(np.matmul(np.inv(intrinsicParameters),homographyMatrix[:,1]))
    lamda = ((magnitude1 + magnitude2)/2)**-1
    r1 = lamda*B[:,0]
    r2 = lamda*B[:,1]
    r3 = np.cross(r1, r2)
    t =  lamda*B[:,2]

    projection_matrix = np.matmul(intrinsicParameters, np.stack(r1,r2,r3,t))

    return projection_matrix

def processFrame(frame):
    pf = preprocessing(frame)
    tag = detectArTag(pf,frame)


def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    edges = cv2.Canny(blur,100,200)
    # _,thresh = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
    return edges 


cap = cv2.VideoCapture('./data/Video_dataset/Tag0.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    processFrame(frame)

    cv2.imshow('AR Tag',frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
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





