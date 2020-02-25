import numpy as np
import cv2
import matplotlib.pyplot as plt

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



# To compute homography between world and camera coordinates
def homography(world_coordinates, pixel_coodinates):
    print(np.shape(world_coordinates))
    xw1 = world_coordinates[0,0]
    xw2 = world_coordinates[1,0]
    xw3 = world_coordinates[2,0]
    xw4 = world_coordinates[3,0]
    yw1 = world_coordinates[0,1]
    yw2 = world_coordinates[1,1]
    yw3 = world_coordinates[2,1]
    yw4 = world_coordinates[3,1]

    xc1 = pixel_coodinates[0,0]
    xc2 = pixel_coodinates[1,0]
    xc3 = pixel_coodinates[2,0]
    xc4 = pixel_coodinates[3,0]
    yc1 = pixel_coodinates[0,1]
    yc2 = pixel_coodinates[1,1]
    yc3 = pixel_coodinates[2,1]
    yc4 = pixel_coodinates[3,1]

    A = np.array([[-xw1, -yw1, -1, 0, 0, 0, xw1 * xc1, yw1 * xc1, xc1],
              [0, 0, 0, -xw1, -yw1, -1, xw1 * yc1, yw1 * yc1, yc1],
              [-xw2, -yw2, -1, 0, 0, 0, xw2 * xc2, yw2 * xc2, xc2],
              [0, 0, 0, -xw2, -yw2, -1, xw2 * yc2, yw2 * yc2, yc2],
              [-xw3, -yw3, -1, 0, 0, 0, xw3 * xc3, yw3 * xc3, xc3],
              [0, 0, 0, -xw3, -yw3, -1, xw3 * yc3, yw3 * yc3, yc3],
              [-xw4, -yw4, -1, 0, 0, 0, xw4 * xc4, yw4 * xc4, xc4],
              [0, 0, 0, -xw4, -yw4, -1, xw4 * yc4, yw4 * yc4, yc4], ])

    [u, sigma, v] =  np.linalg.svd(A)
   
    homography_matrix = v[8,:]/v[8,8]
    homography_matrix = np.reshape(homography_matrix, (3,3))

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


def retrieveInfo(warpedtag):
    _,thresh = cv2.threshold(warpedtag,220,255,cv2.THRESH_BINARY)
    print(thresh)

    region1 = 1 if(thresh[82,83][2] == 255) else 0
    region2 = 1 if(thresh[112,82][2] == 255) else 0
    region3 = 1 if(thresh[84,116][2] == 255) else 0
    region4 = 1 if(thresh[112,115][2] == 255) else 0
    cv2.putText(thresh,'1',(82,83),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)
    cv2.putText(thresh,'2',(110,82),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)
    cv2.putText(thresh,'3',(82,116),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)
    cv2.putText(thresh,'4',(110,115),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)
    cv2.imshow("perspective",thresh)

    if(thresh[60,56][2] == 255): #upper left corner
        return [region4,region3,region1,region2]

    elif(thresh[137,55][2] == 255):#upper right corner
        return [region2,region4,region3,region1]

    elif(thresh[60,140][2]==255): #lower left corner
        return [region1,region2,region4,region3]

    else:
        # (thresh[137,136][2]==255):#lower right corner
        return [region3,region1,region2,region4]



def warpFrame(frame,H,dsize,dc):
    result = cv2.warpPerspective(frame,H,(200,200))

    minPt = (np.amin(dc,axis=0)).astype(int)
    maxPt = (np.amax(dc,axis=0)).astype(int)

    result = np.zeros((dsize[0],dsize[1],frame.shape[2]),dtype ='float32')
    for i in range(minPt[0],maxPt[0]+1):
        for j in range(minPt[1],maxPt[1]+1):
            imageCoor = H.dot([i,j,1])
            hi,hj,_= (imageCoor/imageCoor[2]).astype(int)
            # cv2.circle(frame,(i,j),5,(0,255,0),5)
            if(hi>=0 and hi< dsize[0] and hj>=0 and hj<dsize[1]):
                print(i,j)
                result[hi,hj] = frame[j,i]
    
    result = cv2.warpPerspective(frame,H,(200,200))
    return result



def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    edges = cv2.Canny(blur,100,200)
    return edges 




