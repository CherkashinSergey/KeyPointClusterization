import os
import sys
import cv2
import numpy

def loadDir(dirName):
    files = os.listdir(dirName)
    fnames = []
    for f in files:
        if not f.lower().endswith('.jpg'):
            continue
        fileName = dirName + '\\' + f
        fnames.append(fileName)
    return fnames

#Resizing image to size
def fitImage(img, size = None):
    if size is None:
        global IMAGE_MIN_SIZE
        size = IMAGE_MIN_SIZE
    min_side = min(img.shape[:2])
    cf = float(size) / min_side
    newSize = (int(cf * img.shape[1]), int(cf * img.shape[0]))
    img = cv2.resize(img, newSize)
    return img

def buildKeyPoints(sampleFileList, mode = 'sift', n_KP = None, hessian = None, count = False):
    kpCount = 0
    keyPoints = []
    #descriptors = []
    #imageSizes = []
    if mode == 'sift':
        if n_KP is None: detector = cv2.SIFT()
        else: detector = cv2.SIFT(nfeatures = n_KP)
    else:
        if hessian is None: detector = cv2.SURF()
        else: detector = cv2.SURF(hessianThreshold = hessian)
    
    for i in range (len(sampleFileList)):
        sys.stdout.write('Building keypoints for image ' + str(i+1) + ' of ' + str(len(sampleFileList)) + ' (' + (os.path.split(os.path.dirname(sampleFileList[i])))[1] +')...\r')
        file = sampleFileList[i]
        #logWrite('Buildind descriptors for image ' + file + '\n')
        #Building keypionts, descriptors,
        img = cv2.imread(file)              #read image
        img = fitImage(img)                 #resize image
        #imageSizes.append((img.shape[0],img.shape[1]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #kp, des = sift.detectAndCompute(gray,None) #build keypoints and descriptors on gray image
        kp = detector.detect(gray) #build keypoints and descriptors on gray image
        kpCount += len(kp)
        if kp is None:
            print('Cannot build keypoints to file ' + file)
            kp = []
        keyPoints.append(kp)
    #sys.stdout.write('Building descriptors complete.                              \n')
    if count: return keyPoints, kpCount
    else: return keyPoints

def substractKeypoints(keypoints1, keypoints2, fileNames, saveDir):
    mismatchKP = []
    for index_image in range(len(keypoints1)):
        sys.stdout.write('Comparing image ' + str(index_image + 1) + ' from ' + str(len(keypoints1)) + '\r')
        mismatchCount = 0
        for kp1 in keypoints1[index_image]:
            for kp2 in keypoints2[index_image]:
                if (kp1.pt[0] == kp2.pt[0]) and (kp1.pt[1] == kp2.pt[1]):
                    keypoints1[index_image].remove(kp1)
                    keypoints2[index_image].remove(kp2)
                    break
        mismatchCount = len(keypoints1[index_image])
        #Building image with keypoints
        saveImageWithKP(keypoints1[index_image],fileNames[index_image], saveDir)
        if (len(keypoints2[index_image]) != 0):
            sys.stdout.write('There is '+ str(len(keypoints2[index_image])) + ' left unsubstracted in image ' + fileNames[index_image] + '\n')
        sys.stdout.write('In image ' + fileNames[index_image] + ' there is ' + str(mismatchCount) + ' extra kp.\n')

def saveImageWithKP(keypoints, fileName, saveDir):
    img = cv2.imread(fileName)
    img = fitImage(img)
    img = cv2.drawKeypoints(img,keypoints)
    newName = saveDir + '\\' + fileName
    cv2.imwrite(newName,img)

#####################################################
################      Variables     #################
#####################################################
ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\Test\\'
#ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\'
#ROOT_Dir = 'D:\\ABBYY\\Abbyy photo\\Test0\\'
Dir_A4 = 'A4'
Dir_Card = 'Card'
Dir_Check = 'Check'
Dir_Dual = 'Dual'
Dir_Root = 'Root'
Dir_Single = 'Single'
Dir_SIFT_Difference = 'SiftDiff'
Dir_SURF_Difference = 'SurfDiff'

IMAGE_MIN_SIZE = 700


#####################################################
################ Main functionality #################
#####################################################
os.chdir(ROOT_Dir)
fileNames = loadDir(Dir_A4) + loadDir(Dir_Card) + loadDir(Dir_Check) + loadDir(Dir_Dual) + loadDir(Dir_Root) + loadDir(Dir_Single)
#kp1, count1 = buildKeyPoints(fileNames, mode = 'sift', count = True)
#sys.stdout.write('Keypoints without limitation: ' + str(count1) + '.\n')
#kp2, count2 = buildKeyPoints(fileNames, mode = 'sift', n_KP = 2000, count = True)
#sys.stdout.write('Keypoints with limit 2000: ' + str(count2) + '.\n')
#substractKeypoints(kp1, kp2, fileNames, Dir_SIFT_Difference)

kp1, count1 = buildKeyPoints(fileNames, mode = 'surf', hessian = 500, count = True)
sys.stdout.write('Keypoints without limitation: ' + str(count1) + '.\n')
kp2, count2 = buildKeyPoints(fileNames, mode = 'sift', hessian = 600, count = True)
sys.stdout.write('Keypoints with limitat hess 600: ' + str(count2) + '.\n')
substractKeypoints(kp1, kp2, fileNames, Dir_SURF_Difference)
