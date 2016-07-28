import cv2
import numpy as np
import homography
import imtools
from pylab import *
from PIL import Image


def methodWrap(img):
    img_org = img.copy()
    #img = cv2.medianBlur(img,31)
    img = cv2.GaussianBlur(img,(0,0),2)

    img, cdf = imtools.histeq(img)
    img = np.uint8(img)

    grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.Canny(grayscale, 10, 20)
    thresh = cv2.dilate(thresh,None)

    contours,hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    c, r = 4,2
    coordinates = [[None] * c for i in range(r)]

    for cnt in contours:
        if cv2.contourArea(cnt)>250:  # remove small areas like noise etc
            hull = cv2.convexHull(cnt)    # find the convex hull of contour
            hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
            if len(hull)==4:
                coordinates = hull
                cv2.drawContours(img,[hull],0,(0,255,0),2)


    index1 = 0
    index2 = 0
    index3 = 0
    index4 = 0
    temp = 6000

    for i in range(4):
        if (coordinates[i][0][0] < temp):
            temp = coordinates[i][0][0] 
            index1 = i

    temp = 6000
    for i in range(4):
        if (coordinates[i][0][0] < temp and i != index1):
            temp = coordinates[i][0][0] 
            index2 = i


    if (coordinates[index1][0][1] < coordinates[index2][0][1]):
        coord1 = coordinates[index1][0]
        coord4 = coordinates[index2][0]
    else:
        coord1 = coordinates[index2][0]
        coord4 = coordinates[index1][0]

    for i in range(4):
        if(i != index1 and i != index2):
            index3 = i
            break

    for i in range(4):
        if(i != index1 and i != index2 and i != index3):
            index4 = i
            break

    if (coordinates[index3][0][1] < coordinates[index4][0][1]):
        coord2 = coordinates[index3][0]
        coord3 = coordinates[index4][0]
    else:
        coord2 = coordinates[index4][0]
        coord3 = coordinates[index3][0]

    coor1 = np.append(coord1, 1)
    coor2 = np.append(coord2, 1)
    coor3 = np.append(coord3, 1)
    coor4 = np.append(coord4, 1)

   # print coor1, coor2, coor3, coor4

    w = coor2[0] - coor1[0]
    h = coor4[1] - coor1[1]
    
    fp = np.array([coor1, coor2, coor3, coor4]).T
    tp = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    H = homography.H_from_points(tp, fp)
    im_g = imtools.Htransform(img_org, H, (h, w))

    return im_g