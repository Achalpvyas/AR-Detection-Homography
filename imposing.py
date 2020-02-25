import cv2
import numpy as np
from detection import processFrame
from detection import homography
from detection import projectionMatrix

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
        if len(processFrame(frame)) != 0:
            # Corners of tag and ouside 4 corners
            tagcorners = processFrame(frame)
            if len(tagcorners) != 0:
                # Corners of the AR tag
                tagarray = np.array(tagcorners[1], dtype=float)
                # Homography between Lena image corners and tag image corners
                h = homography(lenaCorners, tagarray)
                # Warping Lena and AR tag
                img = warping(imgframe, h, tagarray, frame)
                # Projecting Lena onto AR tag
                projection = projectionMatrix(h)
                # Appending frames of projected image to form new video
                newFrames.append(img)
        cv2.imshow('AR Tag',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(newFrames)
    return newFrames, size


def perspective_for_tag(ctr,image):
    dst1 = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]], dtype = "float32")

    M1,status = cv2.findHomography(ctr[0], dst1)
    warp1 = cv2.warpPerspective(image.copy(), M1, (100,100))
    warp2=cv2.medianBlur(warp1,3)
    #warp2= warp1-warp1_5

    tag_image=cv2.resize(warp2, dsize=None, fx=0.08, fy=0.08)
    return tag_image,warp2


# Funcion to be modified
def warping(frame, homography, tagcorners, image):

    temp = cv2.warpPerspective(frame, homography,(image.shape[1],image.shape[0]))
    cv2.fillConvexPoly(image,tagcorners.astype(int),0,16)
    image = image + temp
    print(image)
    return image

# Function to convert frames to video
def video(img_array,size):
    video=cv2.VideoWriter('video_Tag.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    #print(np.shape(img_array))
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()




# Checking functionality
newFrames, size = imposingimage('Lena.png','Tag0.mp4')
video(newFrames, size)

# cap = cv2.VideoCapture('video_Tag.avi')
# print(cap.isOpened())
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
#     processFrame(frame)

#     cv2.imshow('AR Tag',frame)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()