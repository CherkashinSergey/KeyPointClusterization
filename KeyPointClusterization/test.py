import cv2
import numpy
import sys
import os
import pickle

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
        imageSizes.append((img.shape[0],img.shape[1]))
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

#keyPoints.shape = (n_images * n_keypoints)
def separateDescriptors(keyPoints, descriptors, imageSize, n_cells):
    new_descriptors = [[list() for x in range(n_cells)] for i in range(len(keyPoints))]
    side = int(numpy.sqrt(n_cells))
    for image_index in range(len(keyPoints)):
       for kp_index in range(len(keyPoints[image_index])):
            cell_X = int(numpy.floor(keyPoints[image_index][kp_index].pt[0]/(imageSize[image_index][1]/side)))
            cell_Y = int(numpy.floor(keyPoints[image_index][kp_index].pt[1]/(imageSize[image_index][0]/side)))
            descriptor_new_index = cell_X*side +cell_Y
            (new_descriptors[image_index][descriptor_new_index]).append(descriptors[image_index][kp_index])
    return new_descriptors

def serializeKP(keypoints):
    serial = []
    for image in keypoints:
        temp_image = []
        for kp in image:
            temp_kp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            temp_image.append(temp_kp)
        serial.append(temp_image)
    return serial

def deserializeKP(serial):
    keypoints = []
    for index_image in range(len(serial)):
        image = []
        for index_kp in range(len(serial[index_image])):
            temp_kp = serial[index_image][index_kp]
            kp = cv2.KeyPoint(x=temp_kp[0][0],y=temp_kp[0][1],_size=temp_kp[1], _angle=temp_kp[2], _response=temp_kp[3], _octave=temp_kp[4], _class_id=temp_kp[5])
            image.append(kp)
        keypoints.append(image)
    return keypoints



IMAGE_MIN_SIZE = 700
file = 'D:\\SCherkashin\\test\\DSC_0211.JPG'


fileList = []
fileList.append(file)
keypoints, descriptors, imageSizes = buildDescriptors(fileList)

ser = serializeKP(keypoints)
keypoints = deserializeKP(ser)

separateDescriptors(keypoints, descriptors, imageSizes, 9)

cache = open('11.bin', 'wb')
pickle.dump(ser,cache)
cache.close()

print(str(len(kp)))