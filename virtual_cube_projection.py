import cv2

# Reading a video file.
# cap = cv2.VideoCapture('Tag1.mp4')
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
# avging = cv2.blur(img,(2,2)) 
   
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