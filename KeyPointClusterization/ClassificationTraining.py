import os
import sys
import shutil
import pickle
import cv2
import sklearn
import numpy
from sklearn.cluster import *
from pylab import *
################################################
##########      Functions      #################
################################################

#Implementing of enum
def enum(**enums):
    return type('Enum', (), enums)

#Creates list of strings, containing full list of *.jpg files
def loadDir(dirName):
    files = os.listdir(dirName)
    fnames = []
    for f in files:
        if not f.lower().endswith('.jpg'):
            continue
        fileName = dirName + '\\' + f
        fnames.append(fileName)
    return fnames

#Creates list of tuples(fileName, Class)
def buildAnswers(fileList, dir_A4=None, dir_Card=None, dir_Check=None, dir_Dual=None, dir_Root=None, dir_Single=None):
    #Having fun =)
    if dir_A4 is None:
        global Dir_A4
        dir_A4 = Dir_A4
    if dir_Card is None:
        global Dir_Card
        dir_Card = Dir_Card
    if dir_Check is None:
        global Dir_Check
        dir_Check = Dir_Check
    if dir_Dual is None:
        global Dir_Dual
        dir_Dual = Dir_Dual
    if dir_Root is None:
        global Dir_Root
        dir_Root = Dir_Root
    if dir_Single is None:
        global Dir_Single
        dir_Single = Dir_Single
    
    global Class
    answers = []
    for file in fileList:
        if os.path.dirname(file) == dir_A4:
            answers.append(Class.A4)
        elif os.path.dirname(file) == dir_Card:
            answers.append(Class.CARD)
        elif os.path.dirname(file) == dir_Check:
            answers.append( Class.CHECK)
        elif os.path.dirname(file) == dir_Dual:
            answers.append(Class.DUAL)
        elif os.path.dirname(file) == dir_Root:
            answers.append(Class.ROOT)
        elif os.path.dirname(file) == dir_Single:
            answers.append(Class.SINGLE)
    return answers

#Creates list of tuples(fileName, Class)
def loadDirWithAnswers(dir_A4=None, dir_Card=None, dir_Check=None, dir_Dual=None, dir_Root=None, dir_Single=None):
    #Having fun =)
    if dir_A4 is None:
        global Dir_A4
        dir_A4 = Dir_A4
    if dir_Card is None:
        global Dir_Card
        dir_Card = Dir_Card
    if dir_Check is None:
        global Dir_Check
        dir_Check = Dir_Check
    if dir_Dual is None:
        global Dir_Dual
        dir_Dual = Dir_Dual
    if dir_Root is None:
        global Dir_Root
        dir_Root = Dir_Root
    if dir_Single is None:
        global Dir_Single
        dir_Single = Dir_Single
    
    def singleAnswerList(dir, classtype):
        answer = []
        files = os.listdir(dir)
        for f in files:
            if not f.lower().endswith('.jpg'):
                continue
            fileName = dir + '\\' + f
            answer.append((fileName, classtype))
        return answer

    global Class
    answers = singleAnswerList(dir_A4, Class.A4)
    answers += singleAnswerList(dir_Card, Class.CARD)
    answers += singleAnswerList(dir_Check, Class.CHECK)
    answers += singleAnswerList(dir_Dual, Class.DUAL)
    answers += singleAnswerList(dir_Root, Class.ROOT)
    answers += singleAnswerList(dir_Single, Class.SINGLE)
    
    return answers

#Resizing image to size
#TODO: implement optimal algoritm of image resizing
#def fitImage(img, size = None):
#    if img.shape[1] > 1000:             #resize if big
#        cf = 1000.0 / img.shape[1]                                                            #cf	0.25510204081632654	float
#        newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]))              #newSize	(562, 1000, 3L)	tuple #, img.shape[2]
#        cv2.resize(img, newSize)
#    return img

def fitImage(img, size = None):
    if size is None:
        global IMAGE_MIN_SIZE
        size = IMAGE_MIN_SIZE
    min_side = min(img.shape[:2])
    cf = float(size) / min_side
    newSize = (int(cf * img.shape[1]), int(cf * img.shape[0]))
    img = cv2.resize(img, newSize)
    return img

#Save object "data" to file "fileName"
def saveToCache(data, fileName):
    cache = open(fileName, 'wb')
    pickle.dump(data,cache)
    cache.close()

#Gets data from binary file "fileName"
def loadFromCahe(fileName):
    cache = open(fileName, 'rb')
    #data = zlib.decompress(data)
    data = pickle.load(cache)
    cache.close()
    return data

#Check existing of cache file
def cacheExists(fileName):
    return os.path.exists(fileName)

#Builds histogram from list of features
#TODO: refactoring
def buildHistogram(predictedList, n_clusters):
    hist = [0 for i in range(n_clusters)]
    for i in range(len(predictedList)):
        if predictedList[i] < n_clusters:
            hist[predictedList[i]] += 1
        else:
            print('Indexing exception!!!')
            return
    return hist

#Returns normalized histogram "hist"
#TODO: implement different algorithms of normalizing histograms
def normalizeHistogram(hist):
    divisor = max(hist)
    temp = [0.0 for i in range(len(hist))]
    if divisor == 0:
        return hist
    for i in range(len(hist)):
        temp[i] = float(hist[i]) / divisor
    return temp

def normalizeHistogram2(histIm):
    nbr_bins = len(histIm)
    imhist, bins = histogram(histIm, nbr_bins, normed = True)       

    cdf = imhist.cumsum()
    cdf = 255 * cdf/cdf[-1]

    im2 = interp(histIm, bins[:-1], cdf)

    histIm = hist(im2.flatten(), 256)

    return histIm


#Creates list of tuples(keypoints[],descriptors[])
#def buildDescriptors(sampleFileList):
#    descriptors =[]
#    for i in range (len(sampleFileList)):
#        sys.stdout.write('Buildind descriptors for image ' + str(i+1) + ' of ' + str(len(sampleFileList)) + ' (' + (os.path.split(os.path.dirname(sampleFileList[i])))[1] +')...\r')
#        file = sampleFileList[i]
        
#        #Building keypionts, descriptors,
#        img = cv2.imread(file)              #read image
#        img = fitImage(img)                 #resize image
        
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        sift = cv2.SIFT()
#        kp, des = sift.detectAndCompute(gray,None) #build keypoints and descriptors on gray image
#        if des is None:
#            print('Cannot build descriptors to file ' + file)
#        else:
#            descriptors.append(des)
#    sys.stdout.write('Building descriptors complete.                              \n')
#    return descriptors

def buildDescriptors(sampleFileList):
    keyPoints = []
    descriptors = []
    imageSizes = []
    global log
    for i in range (len(sampleFileList)):
        sys.stdout.write('Buildind descriptors for image ' + str(i+1) + ' of ' + str(len(sampleFileList)) + ' (' + (os.path.split(os.path.dirname(sampleFileList[i])))[1] +')...\r')
        file = sampleFileList[i]
        log.write('Buildind descriptors for image ' + file)
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

#Clasterizes array "samples [samplesDescriptors]" to n clasters
#def clasterize(samples, n_clusters):
#    histogramsList = []
#    #sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters.\n')
#    for i in range(len(samples)):
#        sys.stdout.write('Clasterizing descriptors for image ' + str(i+1) + ' from ' + str(len(samples)) + '...\r')
#        kmeans = KMeans(n_clusters = n_clusters,verbose = False)
#        #TODO: Maybe it makes sense to fit only once? And I don't use descriptors as themselfs. Just their count.
#        kmeans.fit(samples[i])
#        clusters = kmeans.cluster_centers_.squeeze()
#        hist = kmeans.predict(samples[i])
#        hist = buildHistogram(hist, n_clusters)
#        hist = normalizeHistogram(hist)
#        histogramsList.append(hist)
#    sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters complete!              \n')
#    return histogramsList

def clasterizeInCells(samples, n_clusters, n_imageCells):
    histogramsList = [] # shape (n_images, n_clusters*n_imageCells)
    #sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters.\n')
    for index_Image in range(len(samples)):
        sys.stdout.write('Clasterizing descriptors for image ' + str(index_Image+1) + ' from ' + str(len(samples)) + '...\n')
        #TODO: Maybe it makes sense to fit only once? And I don't use descriptors as themselfs. Just their count.
        imageHist = []
        for index_Cell in range(n_imageCells):
            kmeans = KMeans(n_clusters = n_clusters,verbose = False)
#            print ('N samples = ' + str(len(samples[index_Image][index_Cell])))
            kmeans.fit(samples[index_Image][index_Cell])
            clusters = kmeans.cluster_centers_.squeeze()
            hist = kmeans.predict(samples[index_Image][index_Cell])
            hist = buildHistogram(hist, n_clusters)
            hist = normalizeHistogram(hist)
            imageHist += hist
        histogramsList.append(hist)
    sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters complete!              \n')
    return histogramsList

def separate(samples):
    sample = []
    answers = []
    for i in range (len(samples)):
        sample.append(samples[i][0])
        answers.append(samples[i][1])
    return sample, answers

#Separates descriptors by position of it's keypoints on "n_cells" parts.
#Param: keyPoints - array of keypoints. Shape: (n_images, n_keypoints)
#Param: descriptors - array of descriptors. Shape: (n_images, n_keypoints)
#Param: imageSizes - array of tuples of image sizes. Shape: (n_images, (heigh, width))
#Param: n_cells - number of cells of image, in which it should separate descriptors. n_cells = side**2
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

#################################################
################# Constants #####################
#################################################
IMAGE_MIN_SIZE = 700
IMAGE_CELLS_COUNT = 1
MAX_CLUSTER_COUNT = 8
CACHE_FILE_SEPARATION_COUNT = 10

#################################################
############## Global variables #################
#################################################
#Class = enum(A4 = 'A4', CARD = 'Business card', DUAL = 'Dual page', ROOT = 'Book list with root', SINGLE = 'Single book list', CHECK = 'Cash voucher(check)')
Class = enum(A4 = 0, CARD = 1, DUAL = 2, ROOT = 3, SINGLE = 4, CHECK = 5)

Dir_A4 = 'D:\\SCherkashin\\TrainingFolder\\A4'
Dir_Card = 'D:\\SCherkashin\\TrainingFolder\\Card'
Dir_Check = 'D:\\SCherkashin\\TrainingFolder\\Check'
Dir_Dual = 'D:\\SCherkashin\\TrainingFolder\\Dual'
Dir_Root = 'D:\\SCherkashin\\TrainingFolder\\Root'
Dir_Single = 'D:\\SCherkashin\\TrainingFolder\\Single'

#Dir_A4 = 'D:\\SCherkashin\\TrainingFolder\\Test\\A4'
#Dir_Card = 'D:\\SCherkashin\\TrainingFolder\\Test\\Card'
#Dir_Check = 'D:\\SCherkashin\\TrainingFolder\\Test\\Check'
#Dir_Dual = 'D:\\SCherkashin\\TrainingFolder\\Test\\Dual'
#Dir_Root = 'D:\\SCherkashin\\TrainingFolder\\Test\\Root'
#Dir_Single = 'D:\\SCherkashin\\TrainingFolder\\Test\\Single'

#Dir_A4 = 'D:\\ABBYY\\Abbyy photo\\Test\\A4'
#Dir_Card = 'D:\\ABBYY\\Abbyy photo\\Test\\Card'
#Dir_Check = 'D:\\ABBYY\\Abbyy photo\\Test\\Check'
#Dir_Dual = 'D:\\ABBYY\\Abbyy photo\\Test\\Dual'
#Dir_Root = 'D:\\ABBYY\\Abbyy photo\\Test\\Root'
#Dir_Single = 'D:\\ABBYY\\Abbyy photo\\Test\\Single'

CacheFile_Descriptors = 'D:\\SCherkashin\\TrainingFolder\\descriptors.bin'
CacheFile_Clusters = 'D:\\SCherkashin\\TrainingFolder\\clusters.bin'
CacheFile_Classifier = 'D:\\SCherkashin\\TrainingFolder\\classifier.bin'
LogFile = 'D:\\SCherkashin\\TrainingFolder\\log.txt'




#################################################
############# Main functionality ################
#################################################


log = open(LogFile, 'w')
#TODO: BUILDING SAMPLES (descriptors)
if cacheExists(CacheFile_Descriptors):
    sys.stdout.write('Loading descriptors from cache.\n')
    #samplesDescriptors, answers = loadFromCahe(CacheFile_Descriptors)
    serailizedKeyPoints, samplesDescriptors, samplesImageSizes, answers = loadFromCahe(CacheFile_Descriptors)
    samplesKeyPoints = deserializeKP(serailizedKeyPoints)

else:
    sys.stdout.write('Preparing image descriptors.\n')
    samplesFiles = loadDir(Dir_A4) + loadDir(Dir_Card) + loadDir(Dir_Check) + loadDir(Dir_Dual) + loadDir(Dir_Root) + loadDir(Dir_Single)
    answers = buildAnswers(samplesFiles)
    #samplesDescriptors = buildDescriptors(samplesFiles)
    #data = samplesDescriptors, answers
    samplesKeyPoints, samplesDescriptors, samplesImageSizes = buildDescriptors(samplesFiles)
    data = serializeKP(samplesKeyPoints), samplesDescriptors, samplesImageSizes, answers
    sys.stdout.write('Saving image descriptors to cache.\n')
    saveToCache(data,CacheFile_Descriptors)


#Separating images to different cells count
samplesSeparatedDescriptors = separateDescriptors(samplesKeyPoints, samplesDescriptors, samplesImageSizes, IMAGE_CELLS_COUNT)

for i in range(MAX_CLUSTER_COUNT):
    n_clusters = 2**i
    #TODO: creating histograms of images (later - of image perts)
    if cacheExists(CacheFile_Clusters + str(n_clusters)):
        sys.stdout.write('Loading clusters from cache.\n')
        samplesHistogram = loadFromCahe(CacheFile_Clusters + str(n_clusters))
    else:
        sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clusters.\n')
        #samplesHistogram = clasterize(samplesDescriptors,8)
        samplesHistogram = clasterizeInCells(samplesSeparatedDescriptors, n_clusters, IMAGE_CELLS_COUNT)
        sys.stdout.write('Saving clasterization histograms to cache.\n')
        saveToCache(samplesHistogram,CacheFile_Clusters + str(n_clusters))
    

    #TODO: TRAIN CLASSIFIERS
    if cacheExists(CacheFile_Classifier + str(n_clusters)):
        sys.stdout.write('Loading classifier from cache.\n')
        l_svm, trainSamples, testSamples, trainAnswers, testAnswers = loadFromCahe(CacheFile_Classifier + str(n_clusters))
    else:
        sys.stdout.write('Training classifier.\n')
        l_svm = sklearn.svm.LinearSVC()                 #Creating classifier object
        trainSamples, testSamples, trainAnswers, testAnswers = sklearn.cross_validation.train_test_split(samplesHistogram, answers)
        l_svm.fit(trainSamples, trainAnswers)            #training classifier
        sys.stdout.write('Saving classifier to cache.\n')
        data = l_svm, trainSamples, testSamples, trainAnswers, testAnswers
        saveToCache(data, CacheFile_Classifier + str(n_clusters))

    #l_svm = sklearn.svm.LinearSVC()                 #Creating classifier object
    #trainSamples, testSamples, trainAnswers, testAnswers = sklearn.cross_validation.train_test_split(samplesHistogram, answers)
    #l_svm.fit(trainSamples, trainAnswers)            #training classifier


    #TODO: use cross-validation
    #TODO: use different classifiers

    #TODO: RUN VALIDATION TEST




    #TODO: CHECK ACCURACY
    sys.stdout.write('Testing accuracy of classifier.\n')
    accuracy = l_svm.score(testSamples, testAnswers)


    log.write('Total train files: ' + str(len(trainSamples)) + '\n')
    log.write('Total test files: ' + str(len(testSamples)) + '\n')
    log.write('Accuracy ' + str(accuracy) + ' %.\n')
    

    sys.stdout.write('Total train files: ' + str(len(trainSamples)) + '\n')
    sys.stdout.write('Total test files: ' + str(len(trainSamples)) + '\n')
    sys.stdout.write('Accuracy ' + str(accuracy) + ' %.\n')
log.close()