import cv2
import numpy
import sys
import os

def fitImage(img, size = None):
    if size is None:
        global IMAGE_MIN_SIZE
        size = IMAGE_MIN_SIZE
    
    min_side = min(img.shape[:2])
    cf = float(size) / min_side                                                           #cf	0.25510204081632654	float
    newSize = (int(cf * img.shape[1]), int(cf * img.shape[0]))              #newSize	(562, 1000, 3L)	tuple #, img.shape[2]
    cv2.resize(img, newSize)
    return img

def buildDescriptors(sampleFileList):
    keyPoints = []
    descriptors = []
    imageSizes = []
    for i in range (len(sampleFileList)):
        sys.stdout.write('Buildind descriptors for image ' + str(i+1) + ' of ' + str(len(sampleFileList)) + ' (' + (os.path.split(os.path.dirname(sampleFileList[i])))[1] +')...\r')
        file = sampleFileList[i]
        
        #Building keypionts, descriptors,
        img = cv2.imread(file)              #read image
        img = fitImage(img)                 #resize image
        imageSizes.append(img.shape[:2])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray,None) #build keypoints and descriptors on gray image
        if des is None:
            print('Cannot build descriptors to file ' + file)
        else:
            descriptors.append(des)
            keyPoints.append(kp)
    sys.stdout.write('Building descriptors complete.                              \n')
    return keyPoints, descriptors, imageSizes

def separateDescriptors(keyPoints, descriptors, imageSize, n_cells):
    new_descriptors = [0 for image in range (n_cells)]
    side = int(numpy.sqrt(n_cells))
    for image in range(len(keyPoints)):
        for kp_index in range(len(keyPoints[image])):
            #TODO: remake indexing
            cell_X = numpy.floor(keyPoints[image].pt[1]/(imageSize[image][0]/side))
            cell_Y = numpy.floor(keyPoints[image].pt[0]/(imageSize[image][1]/side))
            index = cell_X*side +cell_Y
            new_descriptors[index].append(descriptors[image])
    return new_descriptors


IMAGE_MIN_SIZE = 700
file = 'C:\\Users\\Zloj\\Source\\Repos\\KeyPointClusterization\\KeyPointClusterization\\1.jpg'
#img = cv2.imread(file) #(1840L, 3264L, 3L)
#min_size = min(img.shape[0], img.shape[1])
#SIZE = 700.0
#divizor = SIZE/min_size
#newSize = (int(img.shape[1] * divizor), int(img.shape[0] * divizor))
#img2 = cv2.resize(img, newSize,interpolation = cv2.INTER_CUBIC)

#gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#sift = cv2.SIFT()
#kp, des = sift.detectAndCompute(gray,None) #build keypoints and descriptors on gray image

fileList = []
fileList.append(file)
keypoints, descriptors, imageSizes = buildDescriptors(fileList)



separateDescriptors(keypoints, descriptors, imageSizes, 9)

print(str(len(kp)))