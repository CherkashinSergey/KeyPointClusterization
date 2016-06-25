import os
import sys
import shutil
import pickle
import cv2
import sklearn
import numpy
from sklearn.cluster import *
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

#Creates list of tuples(keypoints[],descriptors[], class)
#def buildDescriptors(sampleFileList):
#    descriptors =[]
#    for i in range (len(sampleFileList)):
#        sys.stdout.write('Buildind descriptors for image ' + str(i+1) + ' of ' + str(len(sampleFileList)) + ' (' + sampleFileList[i][1] +')...\r')
#        file = sampleFileList[i][0]
        
#        #Building keypionts, descriptors,
#        img = cv2.imread(file)              #read image
#        img = fitImage(img)                 #resize image
        
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        sift = cv2.SIFT()
#        kp, des = sift.detectAndCompute(gray,None) #build keypoints and descriptors on gray image
#        if des is None:
#            print('Cannot build descriptors to file ' + file)
#        else:
#            descriptors.append((des,sampleFileList[i][1]))
#    sys.stdout.write('Building descriptors complete.                              \n')
#    return descriptors

#Resizing image to size
#TODO: implement optimal algoritm of image resizing
def fitImage(img, size = None):
    if img.shape[1] > 1000:             #resize if big
        cf = 1000.0 / img.shape[1]                                                            #cf	0.25510204081632654	float
        newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]))              #newSize	(562, 1000, 3L)	tuple #, img.shape[2]
        cv2.resize(img, newSize)
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

#Clasterizes array "samples [samplesDescriptors][answers]" to n clasters
#def clasterize(samples, n_clusters):
#    histogramsList = []
#    #sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters.\n')
#    for i in range(len(samples)):
#        sys.stdout.write('Clasterizing descriptors for image ' + str(i+1) + ' from ' + str(len(samples)) + '...\r')
#        kmeans = KMeans(n_clusters = n_clusters,verbose = False)
#        kmeans.fit(samples[i][0])
#        clusters = kmeans.cluster_centers_.squeeze()
#        hist = kmeans.predict(samples[i][0])
#        hist = buildHistogram(hist, n_clusters)
#        hist = normalizeHistogram(hist)
#        histogramsList.append((hist, samples[i][1]))
#    sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters complete!              \n')
#    return histogramsList


########################################################
###### functionality without answers ###################
########################################################

#Creates list of tuples(keypoints[],descriptors[], class)
def buildDescriptors(sampleFileList):
    descriptors =[]
    for i in range (len(sampleFileList)):
        sys.stdout.write('Buildind descriptors for image ' + str(i+1) + ' of ' + str(len(sampleFileList)) + ' (' + (os.path.split(os.path.dirname(sampleFileList[i])))[1] +')...\r')
        file = sampleFileList[i]
        
        #Building keypionts, descriptors,
        img = cv2.imread(file)              #read image
        img = fitImage(img)                 #resize image
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray,None) #build keypoints and descriptors on gray image
        if des is None:
            print('Cannot build descriptors to file ' + file)
        else:
            descriptors.append(des)
    sys.stdout.write('Building descriptors complete.                              \n')
    return descriptors

#Clasterizes array "samples [samplesDescriptors]" to n clasters
def clasterize(samples, n_clusters):
    histogramsList = []
    #sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters.\n')
    for i in range(len(samples)):
        sys.stdout.write('Clasterizing descriptors for image ' + str(i+1) + ' from ' + str(len(samples)) + '...\r')
        kmeans = KMeans(n_clusters = n_clusters,verbose = False)
        kmeans.fit(samples[i])
        clusters = kmeans.cluster_centers_.squeeze()
        hist = kmeans.predict(samples[i])
        hist = buildHistogram(hist, n_clusters)
        hist = normalizeHistogram(hist)
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



#################################################
################# Constants #####################
#################################################
IMAGE_MIN_SIZE = 700



#################################################
############## Global variables #################
#################################################
Class = enum(A4 = 'A4', CARD = 'Business card', DUAL = 'Dual page', ROOT = 'Book list with root', SINGLE = 'Single book list', CHECK = 'Cash voucher(check)')

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

CacheFile_Descriptors = 'descriptors.bin'
CacheFile_Clusters = 'clusters.bin'
CacheFile_Classifier = 'classifier.bin'




#################################################
############# Main functionality ################
#################################################

#TODO: BUILDING SAMPLES (descriptors)
if cacheExists(CacheFile_Descriptors):
    sys.stdout.write('Loading descriptors from cache.\n')
    samplesDescriptors, answers = loadFromCahe(CacheFile_Descriptors)
else:
    sys.stdout.write('Preparing image descriptors.\n')
    samplesFiles = loadDir(Dir_A4) + loadDir(Dir_Card) + loadDir(Dir_Check) + loadDir(Dir_Dual) + loadDir(Dir_Root) + loadDir(Dir_Single)
    answers = buildAnswers(samplesFiles)
    samplesDescriptors = buildDescriptors(samplesFiles)
    data = samplesDescriptors, answers
    sys.stdout.write('Saving image descriptors to cache.\n')
    saveToCache(data,CacheFile_Descriptors)

#TODO: creating histograms of images (later - of image perts)
if cacheExists(CacheFile_Clusters):
    sys.stdout.write('Loading clusters from cache.\n')
    samplesHistogram = loadFromCahe(CacheFile_Clusters)
else:
    sys.stdout.write('Separating descriptors to clusters.\n')
    samplesHistogram = clasterize(samplesDescriptors,8)
    sys.stdout.write('Saving clasterization histograms to cache.\n')
    saveToCache(samplesHistogram,CacheFile_Clusters)
    

#TODO: TRAIN CLASSIFIERS
if cacheExists(CacheFile_Classifier):
    sys.stdout.write('Loading classifier from cache.\n')
    l_svm, trainSamples, testSamples, trainAnswers, testAnswers = loadFromCahe(CacheFile_Classifier)
else:
    sys.stdout.write('Training classifier.\n')
    l_svm = sklearn.svm.LinearSVC()                 #Creating classifier object
    trainSamples, testSamples, trainAnswers, testAnswers = sklearn.cross_validation.train_test_split(samplesHistogram, answers)
    l_svm.fit(trainSamples, trainAnswers)            #training classifier
    sys.stdout.write('Saving classifier to cache.\n')
    data = l_svm, trainSamples, testSamples, trainAnswers, testAnswers
    saveToCache(data, CacheFile_Classifier)

#l_svm = sklearn.svm.LinearSVC()                 #Creating classifier object
#trainSamples, testSamples, trainAnswers, testAnswers = sklearn.cross_validation.train_test_split(samplesHistogram, answers)
#l_svm.fit(trainSamples, trainAnswers)            #training classifier


#TODO: use cross-validation
#TODO: use different classifiers

#TODO: RUN VALIDATION TEST




#TODO: CHECK ACCURACY
sys.stdout.write('Testing accuracy of classifier.\n')
accuracy = l_svm.score(testSamples, testAnswers)

log = open('log.txt', 'w')
log.write('Total train files: ' + str(len(trainSamples)) + '\n')
log.write('Total test files: ' + str(len(trainSamples)) + '\n')
log.write('Accuracy ' + str(accuracy) + ' %.\n')
log.close()

sys.stdout.write('Total train files: ' + str(len(trainSamples)) + '\n')
sys.stdout.write('Total test files: ' + str(len(trainSamples)) + '\n')
sys.stdout.write('Accuracy ' + str(accuracy) + ' %.\n')