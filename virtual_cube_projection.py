import cv2
import numpy as np
from detection import processFrame
# Reading a video file.
# cap = cv2.VideoCapture('Tag0.mp4')
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     #print(frame)
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()




# Filtering an image
# Averaging techniques
# importing opencv CV2 module 
# import cv2  
  
# # bat.jpg is the batman image. 
# img = cv2.imread('Lena.png') 
   
# # make sure that you have saved it in the same folder 
# # Averaging 
# # You can change the kernel size as you want 
# avging = cv2.blur(img,(10,10)) 
   
# cv2.imshow('Averaging',avging) 
# cv2.waitKey(0) 
  
# # Gaussian Blurring 
# # Again, you can change the kernel size 
# gausBlur = cv2.GaussianBlur(img, (3,3),0)  
# cv2.imshow('Gaussian Blurring', gausBlur) 
# cv2.waitKey(0) 
  
# # Median blurring 
# medBlur = cv2.medianBlur(img,5) 
# cv2.imshow('Media Blurring', medBlur) 
# cv2.waitKey(0) 
  
# # Bilateral Filtering 
# bilFilter = cv2.bilateralFilter(img,9,75,75) 
# cv2.imshow('Bilateral Filtering', bilFilter) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 




# # Python program to explain cv2.cvtColor() method  
   
# # importing cv2  
   
# # Reading an image in default mode 
# src = cv2.imread('Lena.png') 
   
# # Window name in which image is displayed 
# window_name = 'Image'
  
# # Using cv2.cvtColor() method 
# # Using cv2.COLOR_BGR2GRAY color space 
# # conversion code 

# ###################################
# #       Gray Color space COLOR_BGR2GRAY
# ################################
# #image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY ) 


# ###################################
# #       Gray Color space COLOR_BGR2HSV
# ###################################
# image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV ) 
  
# # Displaying the image  
# cv2.imshow(window_name, image) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 





# # Python programe to illustrate  
# # simple thresholding type on an image 
      
# # organizing imports 
# import numpy as np  
  
# # path to input image is specified and   
# # image is loaded with imread command  
# image1 = cv2.imread('Lena.png')  
  
# # cv2.cvtColor is applied over the 
# # image input with applied parameters 
# # to convert the image in grayscale  
# img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
# medBlur = cv2.GaussianBlur(img, (3,3),0) 
# # applying different thresholding  
# # techniques on the input image 
# # all pixels value above 120 will  
# # be set to 255 
# ret, thresh1 = cv2.threshold(medBlur, 120, 255, cv2.THRESH_BINARY) 
# ret, thresh2 = cv2.threshold(medBlur, 120, 255, cv2.THRESH_BINARY_INV) 
# ret, thresh3 = cv2.threshold(medBlur, 120, 255, cv2.THRESH_TRUNC) 
# ret, thresh4 = cv2.threshold(medBlur, 120, 255, cv2.THRESH_TOZERO) 
# ret, thresh5 = cv2.threshold(medBlur, 120, 255, cv2.THRESH_TOZERO_INV) 
  
# # the window showing output images 
# # with the corresponding thresholding  
# # techniques applied to the input images 
# cv2.imshow('Binary Threshold', thresh1) 
# cv2.imshow('Binary Threshold Inverted', thresh2) 
# cv2.imshow('Truncated Threshold', thresh3) 
# cv2.imshow('Set to 0', thresh4) 
# cv2.imshow('Set to 0 Inverted', thresh5) 
    
# # De-allocate any associated memory usage   
# if cv2.waitKey(0) & 0xff == 27:  
#     cv2.destroyAllWindows()  


# import numpy as np 
  
# # Let's load a simple image with 3 black squares 
# image = cv2.imread('Lena.png') 
# cv2.waitKey(0) 
  
# # Grayscale 
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# # Find Canny edges 
# edged = cv2.Canny(gray, 30 , 200) 
# cv2.waitKey(0) 

# edged = cv2.Canny(gray, 100 , 200) 
# cv2.waitKey(0)
  
# # Finding Contours 
# # Use a copy of the image e.g. edged.copy() 
# # since findContours alters the image 
# contours, hierarchy = cv2.findContours(edged,  
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
# print(hierarchy[0])
# cv2.imshow('Canny Edges After Contouring', edged) 
# cv2.waitKey(0) 
  
# print("Number of Contours found = " + str(len(contours))) 
  
# # Draw all contours 
# # -1 signifies drawing all contours 
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
  
# cv2.imshow('Contours', image) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 


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

    homography_matrix = v[:,8]/v[8,8]
    print(homography_matrix)
    homography_matrix = np.reshape(homography_matrix, (3,3))

    return homography_matrix

# For camera pose estimation
def projectionMatrix(homographyMatrix):
    intrinsicParameters =np.array([[1406.08415449821,0,0],
                                  [2.20679787308599, 1417.99930662800,0],
                                  [1014.13643417416, 566.347754321696,1]])

    intrinsicParameters = np.transpose(intrinsicParameters)

    B = np.matmul(np.linalg.inv(intrinsicParameters), homographyMatrix)
    if np.linalg.det(B) < 0:
        B = -1*B
    
    magnitude1 = np.linalg.norm(np.matmul(np.linalg.inv(intrinsicParameters),homographyMatrix[:,0]))
    magnitude2 = np.linalg.norm(np.matmul(np.linalg.inv(intrinsicParameters),homographyMatrix[:,1]))
    lamda = ((magnitude1 + magnitude2)/2)**-1
    r1 = lamda*B[:,0]
    r2 = lamda*B[:,1]
    r3 = np.cross(r1, r2)
    t =  lamda*B[:,2]

    projection_matrix = np.matmul(intrinsicParameters, np.stack((r1,r2,r3,t)).T)

    return projection_matrix

def Cube3D(proj_mat,image):
    axis = np.float32([[0,0,0,1],[0,512,0,1],[512,512,0,1],[512,0,0,1],[0,0,-512,1],[0,512,-512,1],[512,512,-512,1],[512,0,-512,1]])
    Proj= np.matmul(axis,proj_mat.T)
    # Normalize the matrix
    Norm1 = np.divide(Proj[0],Proj[0][2])
    Norm2 = np.divide(Proj[1],Proj[1][2])
    Norm3 = np.divide(Proj[2],Proj[2][2])
    Norm4 = np.divide(Proj[3],Proj[3][2])
    Norm5 = np.divide(Proj[4],Proj[4][2])
    Norm6 = np.divide(Proj[5],Proj[5][2])
    Norm7 = np.divide(Proj[6],Proj[6][2])
    Norm8 = np.divide(Proj[7],Proj[7][2])

    points = np.vstack((Norm1,Norm2,Norm3,Norm4,Norm5,Norm6,Norm7,Norm8))
    final_2d=np.delete(points,2, axis=1)
    draw(image,final_2d)
    return image

def draw(img, imgpts):
    imgpts = np.int32(imgpts)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(255,0,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,150,150),3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    cv2.waitKey(0)
    return img




img = cv2.imread('ref_marker.png')
#corners = processFrame(img)

cap = cv2.VideoCapture('Tag1.mp4')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    src=cv2.imread('lena.jpg')
    # img = cv2.imread(frame)
    if len(processFrame(frame)) != 0:
        corners = processFrame(frame)
        if len(corners) == 0:
            continue
        #pts_dst = np.array(corners[1])
        pts_src = np.array([[511, 511],[0,511],[0,0],[511, 0]],dtype=float)
        h = homography(pts_src,corners[1])
        proj_mat = projectionMatrix(h)
        image=Cube3D(proj_mat,frame)
        cv2.imshow('cube',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        continue

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()








