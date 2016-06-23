import os
import sys
import shutil
import pickle
import copy
import cv2
import sklearn
import numpy
from sklearn.cluster import *

GoodDir = 'D:\\SCherkashin\\DocsPhoto\\Good'
BadDir = 'D:\\SCherkashin\\DocsPhoto\\Bad'
TestDir = 'D:\\SCherkashin\\test'
CacheFile = 'descriptors.bin'
ClustersFile = 'clusters.bin'
CLUSTER_RANGE = 8

def loadDir(dirName):
    files = os.listdir(dirName)
    fnames = []
    for f in files:
        if not f.lower().endswith('.jpg'):
            continue
        fileName = dirName + '\\' + f
        fnames.append(fileName)
    return fnames

def reshape(desc):
    desc1 = []
    for i in range (len(desc)):
        #if len(desc[i]) != len(desc[0]):
        #        print('In dim2 ' + str(i) + ' len is ' + str(len(desc[i])) + '. Should be: ' + str(len(desc[0])))
        for j in range (len(desc[i])):
            desc1.append(desc[i][j])
        #    if len(desc[i][j]) != len(desc[0][0]):
        #        print('In dim3 ' + str(i) + ' len is ' + str(len(desc[i][j]))  + '. Should be: ' + str(len(desc[0][0])))
    return desc1

def getDescriptors(fileName):
    img = cv2.imread(fileName)          #read image
    if img.shape[1] > 1000:             #resize if big
        cf = 1000.0 / img.shape[1]                                                            #cf	0.25510204081632654	float
        newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]))              #newSize	(562, 1000, 3L)	tuple #, img.shape[2]
        cv2.resize(img, newSize)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(img,None)
    return des

def buildDescriptors(fileList):
    descriptors =[]
    n = 1
    for file in fileList:
        sys.stdout.write('Buildind descriptors for image ' + str(n) + ' of ' + str(len(fileList)) + '...\r')
        des = getDescriptors(file)
        if des is None:
            print('Cannot build descriptors to file ' + file)
            os.remove(file)
        else:
            descriptors.append(des)
        n += 1
    return descriptors

def buildHistogram(predictedList, n_clusters):
    hist = [0 for i in range(n_clusters)]
    for i in range(len(predictedList)):
        if predictedList[i] < n_clusters:
            hist[predictedList[i]] += 1
        else:
            print('Indexing exception!!!')
            return
    return hist

def normalizeHistogram(hist):
    divisor = max(hist)
    temp = [0.0 for i in range(len(hist))]
    if divisor == 0:
        return hist
    for i in range(len(hist)):
        temp[i] = float(hist[i]) / divisor
    return temp



sys.stdout.write('Searching for cache...\r')
if not(os.path.isfile(ClustersFile)):
    if not(os.path.isfile(CacheFile)):
        print('Opening files')
        #files = loadDir(GoodDir) + loadDir(BadDir)
        files = loadDir(TestDir)
        print('Building descriptors')
        desc = buildDescriptors(files)
        #desc = numpy.reshape((n_images * n_key_points, n_features))
        desc = reshape(desc)
    
        #creating cache
        sys.stdout.write('Saving descriptors to cache...\r')
        data = desc
        cache = open(CacheFile, 'wb')
        pickle.dump(data,cache)
        cache.close()
    else:
        sys.stdout.write('Loading descriptors from cache...\r')
        cache = open(CacheFile, 'rb')
        #data = zlib.decompress(data)
        data = pickle.load(cache)
        cache.close()
        desc = data

    print('Computing clusters')
    clusterCenters_Range = []
    kmeans_Range = []
    hist_Range = []
    for i in range(3, CLUSTER_RANGE):
        n_clusters = 2 ** i
        sys.stdout.write('Building clusters in range ' + str(n_clusters) +'...\r')
        kmeans = KMeans(n_clusters = n_clusters,verbose = False)
        kmeans.fit(desc)
        clusters = kmeans.cluster_centers_.squeeze()
        hist = kmeans.predict(desc)
        
        hist = buildHistogram(hist, n_clusters)
        hist = normalizeHistogram(hist)
        
        hist_Range.append(hist)
        clusterCenters_Range.append(clusters)
        print('Clusters count: ' + str(n_clusters) + '. Shape of desc: ' + str(numpy.shape(desc)) + '. Shape of clusters: ' + str(numpy.shape(clusters)))
        
    
    sys.stdout.write('Saving descriptors to cache...\r')
    data = kmeans_Range, clusterCenters_Range, hist_Range
    cache = open(ClustersFile, 'wb')
    pickle.dump(data,cache)
    cache.close() 

else:
    sys.stdout.write('Loading clusters from cache...\r')
    cache = open(ClustersFile, 'rb')
    #data = zlib.decompress(data)
    data = pickle.load(cache)
    cache.close()
    kmeans_Range, clusterCenters_Range, hist_Range = data

for i in range(len(hist_Range)):
    print ('Clusters: ' + str(len(clusterCenters_Range[i])))
    print ('Histogram length: ' + str(len(hist_Range[i])))
