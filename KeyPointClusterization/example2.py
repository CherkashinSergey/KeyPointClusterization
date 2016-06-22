import os
import shutil
import sys
import pickle
import zlib
import cv2
import numpy as np
import urllib2
import time
import random
from Queue import Queue
from threading import Thread
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

#######################
BAYES_ALPHA = 0.1
ADA_BOOST_ESTIMATORS = 110
CLUSTER_SEED = 24
CLUSTERS_NUMBER = 1000
VERBOSE = True
Current_Descr = 0
CurrentSample = 0
Removed = False

GoodDir = 'D:\\SCherkashin\\DocsPhoto\\TEST111\\Viz'
BadDir = 'D:\\SCherkashin\\DocsPhoto\\TEST111\\A4'
TestDir = 'D:\\SCherkashin\\DocsPhoto\\TEST111\\Test'
SortedPositiveDir = 'D:\\SCherkashin\\DocsPhoto\\TEST111\\SortedViz'
SortedNegativeDir = 'D:\\SCherkashin\\DocsPhoto\\TEST111\\SortedA4'
CacheFile = 'Cache.bin'
#img = cv2.imread('D:\\SCherkashin\\test\\DSC_0224.JPG')

def loadDir(dirName):
    files = os.listdir(dirName)
    fnames = []
    for f in files:
        if not f.lower().endswith('.jpg'):
            continue
        fileName = dirName + '\\' + f
        fnames.append(fileName)
    return fnames

def addDescriptors(totalDescriptors, samples):
    for sample in samples:
        for descriptor in sample[0]:
            totalDescriptors.append(descriptor)

def makeSamples(files):
    global Removed
    samples = [[]] * len(files)
    n = 0
    for f in files:
        sys.stdout.write('Processing image ' + str(n+1) + ' from ' + str(len(files)) + '...\r')
        des,hist = getFeatures(f)
        if des is None:
            print ('ERROR: failed to load' + f)
            os.remove(f)
            Removed = True
        else:
            samples[n] = (des, hist)
        n += 1
    return samples

def getFeatures(fileName):
    global img
    #fileName = 'D:\\SCherkashin\\test\\DSC_0224.JPG'
    img = cv2.imread(fileName)          #read image
    if img.shape[1] > 1000:             #resize if big
        cf = 1000.0 / img.shape[1]                                                            #cf	0.25510204081632654	float
        newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]), img.shape[2])              #newSize	(562, 1000, 3L)	tuple
        img.resize(newSize)
      
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s = cv2.SIFT(nfeatures = 400)
      
    d = cv2.DescriptorExtractor_create("OpponentSIFT")
    kp = s.detect(gray, None)
    kp, des = d.compute(img, kp)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0],None,[256],[0,256])
        
    return des, hist

#def getFeatures(fileName):
#    #fileName = 'D:\\SCherkashin\\test\\DSC_0224.JPG'
#    img = cv2.imread(fileName)          #read image
#    if img.shape[1] > 1000:             #resize if big
#        cf = 1000.0 / img.shape[1]                                                            #cf	0.25510204081632654	float
#        newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]))              #newSize	(562, 1000, 3L)	tuple #, img.shape[2]
#        cv2.resize(img, newSize)
      
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    s = cv2.SIFT(nfeatures = 400)
      
#    d = cv2.DescriptorExtractor_create("OpponentSIFT")
#    kp = s.detect(gray, None)
#    kp, des = d.compute(img, kp)
    
#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    hist = cv2.calcHist([hsv],[0],None,[256],[0,256])
        
#    return des, hist


def calculteCounts(samples, counts, counts1, clusters):
    global Current_Descr
    global CLUSTERS_NUMBER
    global CurrentSample
    cn = CLUSTERS_NUMBER
    for s in samples:
      currentCounts = {}
      for d in s[0]:
        currentCounts[clusters[Current_Descr]] = currentCounts.get(clusters[Current_Descr], 0) + 1
        Current_Descr += 1
      for clu, cnt in currentCounts.iteritems():
        counts[CurrentSample, clu] = cnt
      for i, histCnt in enumerate(s[1]):
        counts1[CurrentSample, i] = histCnt[0]
      CurrentSample += 1
#######################



#fileName = 'D:\\SCherkashin\\test\\DSC_0211.JPG'
#img = cv2.imread(fileName)          #read image

#if img.shape[1] > 1000:             #resize if big
#  cf = 1000.0 / img.shape[1]                                                            #cf	0.25510204081632654	float
#  newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]), img.shape[2])              #newSize	(562, 1000, 3L)	tuple
#  img.resize(newSize)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #make gray image to find keypoints

#s = cv2.SIFT(nfeatures = 400)

#d = cv2.DescriptorExtractor_create("OpponentSIFT")
#kp = s.detect(gray, None)           #Make keypoints from gray image
#kp, des = d.compute(img, kp)        #Make descriptors for them from color image

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          #Make hsv from image
#dist = cv2.calcHist([hsv],[0],None,[256],[0,256])   #Make histogram 

#Create objects of algorithm executors
tfidf = TfidfTransformer()
tfidf1 = TfidfTransformer()
clf = AdaBoostClassifier(MultinomialNB(alpha = BAYES_ALPHA), n_estimators = ADA_BOOST_ESTIMATORS)
clf1 = AdaBoostClassifier(MultinomialNB(alpha = BAYES_ALPHA), n_estimators = ADA_BOOST_ESTIMATORS)
kmeans = MiniBatchKMeans(n_clusters = CLUSTERS_NUMBER, random_state = CLUSTER_SEED, verbose = VERBOSE)

if not(os.path.isfile(CacheFile)):
    print('Generating cache.')
    print('Counting files')
    positiveFiles = loadDir(GoodDir)
    negativeFiles = loadDir(BadDir)

    print('Processing positive samples')
    positiveSamples = makeSamples(positiveFiles)    #making positive samples
    print('Processing negative samples')
    negativeSamples = makeSamples(negativeFiles)    #making negative samples

    if(Removed):
        Removed = False
        print('Rebuilding samples')
        print('Processing positive samples')
        positiveSamples = makeSamples(positiveFiles)    #making positive samples
        print('Processing negative samples')
        negativeSamples = makeSamples(negativeFiles)    #making negative samples


    totalDescriptors = []
    addDescriptors(totalDescriptors, positiveSamples)   #extracting descroptors from positive samples
    addDescriptors(totalDescriptors, negativeSamples)   #extracting descroptors from negative samples
  
    kmeans.fit(totalDescriptors)                        #Clusterization
    clusters = kmeans.predict(totalDescriptors)         #List of clusters

    totalSamplesNumber = len(negativeSamples) + len(positiveSamples)    #total number of samples
    counts = lil_matrix((totalSamplesNumber, CLUSTERS_NUMBER))          
    counts1 = lil_matrix((totalSamplesNumber, 256))                     
    calculteCounts(positiveSamples, counts, counts1, clusters)          #frequency of each descriptor of each cluster of each image
    calculteCounts(negativeSamples, counts, counts1, clusters)          #frequency of each color of eash image
    counts = csr_matrix(counts)
    counts1 = csr_matrix(counts1)

    _tfidf = tfidf.fit_transform(counts)                                
    _tfidf1 = tfidf1.fit_transform(counts1)
    classes = [True] * len(positiveSamples) + [False] * len(negativeSamples)
    clf.fit(_tfidf, classes)                                            #studuing
    clf1.fit(_tfidf1, classes)

    #data = pickle.dumps((CLUSTERS_NUMBER, kmeans, tfidf, tfidf1, clf, clf1), -1)
    #data = zlib.compress(data)
    data = CLUSTERS_NUMBER, kmeans, tfidf, tfidf1, clf, clf1
    cache = open(CacheFile, 'wb')
    pickle.dump(data,cache)
    cache.close()
else:
    print('Loading cache...')
    cache = open(CacheFile, 'rb')
    #data = zlib.decompress(data)
    data = pickle.load(cache)
    cache.close()
    CLUSTERS_NUMBER, kmeans, tfidf, tfidf1, clf, clf1 = data



##############################
######## PREDICTION ##########
##############################
print('Prediction started.')
testFiles = loadDir(TestDir)
print('Loading test samples')
testSamples = makeSamples(testFiles)
if(Removed):
    print('Rebuild test samples')
    testFiles = loadDir(TestDir)
    print('Loading test samples')
    testSamples = makeSamples(testFiles)

testDescriptors = []
addDescriptors(testDescriptors, testSamples)

testClusters = kmeans.predict(testDescriptors)
testCounts = lil_matrix((len(testSamples), CLUSTERS_NUMBER))
testCounts1 = lil_matrix((len(testSamples), 256))
calculteCounts(testSamples, testCounts, testCounts1, testClusters)
testCounts = csr_matrix(testCounts)
testCounts1 = csr_matrix(testCounts1)

_tfidf = tfidf.transform(testCounts)
_tfidf1 = tfidf1.transform(testCounts1)

weights = clf.predict_log_proba(_tfidf)
weights1 = clf1.predict_log_proba(_tfidf1)
predictions = []
for i in xrange(0, len(weights)):
  w = weights[i][0] - weights[i][1]
  w1 = weights1[i][0] - weights1[i][1]
  pred = w < 0
  pred1 = w1 < 0
  if pred != pred1:
    pred = w + w1 < 0
  predictions.append(pred)

if len(testFiles) == len(predictions):
    log = open('log.txt', 'w')
    for i in range(len(testFiles)):
        log.write(testFiles[i] + ' is ' + str(predictions[i]) + '\n')  #creating log
        if predictions[i]:                                             #sorting files
            shutil.move(testFiles[i],SortedPositiveDir)
        else:
            shutil.move(testFiles[i],SortedNegativeDir)
    log.close()