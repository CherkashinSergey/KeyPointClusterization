import os
import sys
import shutil
import pickle
import cv2
import sklearn
import numpy
from sklearn.cluster import *

GoodDir = 'D:\\SCherkashin\\DocsPhoto\\Good'
BadDir = 'D:\\SCherkashin\\DocsPhoto\\Bad'
TestDir = 'D:\\SCherkashin\\DocsPhoto\\Test'
CacheFile = 'descriptors.bin'
CLUSTER_RANGE = 7

def loadDir(dirName):
    files = os.listdir(dirName)
    fnames = []
    for f in files:
        if not f.lower().endswith('.jpg'):
            continue
        fileName = dirName + '\\' + f
        fnames.append(fileName)
    return fnames

def getDescriptors(fileName):
    img = cv2.imread(fileName)          #read image
    if img.shape[1] > 1000:             #resize if big
        cf = 1000.0 / img.shape[1]                                                            #cf	0.25510204081632654	float
        newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]))              #newSize	(562, 1000, 3L)	tuple #, img.shape[2]
        cv2.resize(img, newSize)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT(nfeatures = 400)
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

if not(os.path.isfile(CacheFile)):
    print('Opening files')
    files = loadDir(GoodDir) + loadDir(BadDir)
    print('Building descriptors')
    desc = buildDescriptors(files)
    #desc = numpy.reshape((n_images * n_key_points, n_features))
    
    
    #creating cache
    data = desc
    cache = open(CacheFile, 'wb')
    pickle.dump(data,cache)
    cache.close()
else:
    print('Loading cache...')
    cache = open(CacheFile, 'rb')
    #data = zlib.decompress(data)
    data = pickle.load(cache)
    cache.close()
    desc = data

print('Computing clusters')
results = []
print(len(desc))
print(len(desc[0]))
print(len(desc[0][0]))
print(numpy.shape(desc))
desc1 = [[]]
for i in range (len(desc)):
    for j in range (len(desc[i])):
        desc1.append(desc[i][j])
print(numpy.shape(desc[i][j]))
print(numpy.shape(desc1))
#desc1 = numpy.reshape(desc,(len(desc) * len(desc[0]) ,len(desc[0][0])))

for i in range(3, CLUSTER_RANGE):
    n_clusters = 2 ** i
    sys.stdout.write('Building clusters in range ' + str(n_clusters) +'...\r')
    kmeans = KMeans(n_clusters = n_clusters,verbose = True)
    clusters = kmeans.fit(desc1)
    results.append(clusters)

for i in range(len(results)):
    print ('Clusters: ' + str(len(results[i])))