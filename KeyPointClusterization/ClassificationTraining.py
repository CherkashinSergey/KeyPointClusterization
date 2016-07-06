import os
import sys
import shutil
import pickle
import cv2
import sklearn
import numpy
from sklearn.cluster import *
from sklearn.ensemble import *
from pylab import *
from exceptions import ValueError
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
def buildDescriptors(sampleFileList):
    keyPoints = []
    descriptors = []
    imageSizes = []
    global log
    for i in range (len(sampleFileList)):
        sys.stdout.write('Buildind descriptors for image ' + str(i+1) + ' of ' + str(len(sampleFileList)) + ' (' + (os.path.split(os.path.dirname(sampleFileList[i])))[1] +')...\r')
        file = sampleFileList[i]
        logWrite('Buildind descriptors for image ' + file + '\n')
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
def clasterizeInCells(samples, n_clusters, n_imageCells):
    histogramsList = [] # shape (n_images, n_clusters*n_imageCells)
    #sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters.\n')
    for index_Image in range(len(samples)):
        sys.stdout.write('Clasterizing descriptors for image ' + str(index_Image+1) + ' from ' + str(len(samples)) + '...\r')
        #TODO: Maybe it makes sense to fit only once? And I don't use descriptors as themselfs. Just their count.
        imageHist = []
        for index_Cell in range(n_imageCells):
            kmeans = KMeans(n_clusters = n_clusters,verbose = False)
            #if (len(samples[index_Image][index_Cell]) <= n_clusters):
            #    sys.stdout.write('n_Clusters = ' + str(n_clusters) + 'n_Samples = ' + str(len(samples[index_Image][index_Cell]) + '. Clusterization impossible! Abort this iteration!\n')
            #    return None
            kmeans.fit(samples[index_Image][index_Cell])
            clusters = kmeans.cluster_centers_.squeeze()
            hist = kmeans.predict(samples[index_Image][index_Cell])
            hist = buildHistogram(hist, n_clusters)
            hist = normalizeHistogram(hist)
            imageHist += hist
        histogramsList.append(hist)
    #sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters complete!              \n')
    return histogramsList

#Clasterizes array "samples (samplesDescriptors,answers)[]" to n clasters
#def clasterizeInCellsWithAnswres(samplesWithAnswers, n_clusters, n_imageCells):
#    histogramsListWithAnswers = [] # shape (n_images, (n_clusters*n_imageCells, n_answers))
#    excludedSamplesCount = 0
#    for index_Image in range(len(samplesWithAnswers)):
#        sys.stdout.write('Clasterizing descriptors for image ' + str(index_Image+1) + ' from ' + str(len(samplesWithAnswers)) + '...\r')
#        #TODO: Maybe it makes sense to fit only once? And I don't use descriptors as themselfs. Just their count.
#        imageHist = []
#        for index_Cell in range(n_imageCells):
#            kmeans = KMeans(n_clusters = n_clusters,verbose = False)
#            if (len(samplesWithAnswers[index_Image][0][index_Cell]) <= n_clusters):
#                excludedSamplesCount += 1
#                break
#            kmeans.fit(samplesWithAnswers[index_Image][0][index_Cell])
#            clusters = kmeans.cluster_centers_.squeeze()
#            hist = kmeans.predict(samplesWithAnswers[index_Image][0][index_Cell])
#            hist = buildHistogram(hist, n_clusters)
#            hist = normalizeHistogram(hist)
#            imageHist += hist
#        else:
#            histogramsListWithAnswers.append((hist, samplesWithAnswers[index_Image][1]))
#    #sys.stdout.write('Separating descriptors to ' + str(n_clusters) + ' clasters complete!              \n')
#    return histogramsListWithAnswers

#def clasterizeInCellsWithAnswres(samplesWithAnswers, n_imageCells, kmeans):
#    histogramsListWithAnswers = [] # shape (n_images, (n_clusters*n_imageCells, n_answers))
#    excludedSamplesCount = 0
#    for index_Image in range(len(samplesWithAnswers)):
#        sys.stdout.write('Clasterizing descriptors for image ' + str(index_Image+1) + ' from ' + str(len(samplesWithAnswers)) + '...\r')
#        imageHist = []
#        for index_Cell in range(n_imageCells):
#            if len(samplesWithAnswers[index_Image][0][index_Cell]) == 0:
#                hist = [0 for x in range(n_clusters)]                                 #in case when some image cell has no descriptors
#            else:
#                hist = kmeans.predict(samplesWithAnswers[index_Image][0][index_Cell])
#                hist = buildHistogram(hist, n_clusters)
#                hist = normalizeHistogram(hist)
#            imageHist += hist
#        else:
#            histogramsListWithAnswers.append((imageHist, samplesWithAnswers[index_Image][1]))
#    return histogramsListWithAnswers

#Clasterizes array "samples (samplesDescriptors,answers)[]" to n clasters
#Param: "stat" if True - collects statistics
def clasterizeInCellsWithAnswres(samplesWithAnswers, n_imageCells, kmeans, stat = False):
    if stat:
        global StatisticsFile
        statFileName = 'CL_' + str(kmeans.n_clusters) + 'CELL_' + str(n_imageCells) + StatisticsFile
        statFile = open(statFileName,'a')
        CSVSeparator = ';'
    histogramsListWithAnswers = [] # shape (n_images, (n_clusters*n_imageCells, n_answers))
    excludedSamplesCount = 0
    for index_Image in range(len(samplesWithAnswers)):
        sys.stdout.write('Clasterizing descriptors for image ' + str(index_Image+1) + ' from ' + str(len(samplesWithAnswers)) + '...\r')
        imageHist = []
        for index_Cell in range(n_imageCells):
            if len(samplesWithAnswers[index_Image][0][index_Cell]) == 0:
                hist = [0 for x in range(n_clusters)]                                   #in case when some image cell has no descriptors
            else:
                hist = kmeans.predict(samplesWithAnswers[index_Image][0][index_Cell])
                hist = buildHistogram(hist, n_clusters)
            if stat:                                                                    #writing statistics
                for value in hist:
                    statFile.write(str(value) + CSVSeparator)
            hist = normalizeHistogram(hist)                                             #Should try normalization on whole imageHist
            imageHist += hist
        histogramsListWithAnswers.append((imageHist, samplesWithAnswers[index_Image][1]))
        if stat:                                                                        #Should be refactored
                statFile.write('\n')
                statFile.flush()
    if stat: statFile.close()
    return histogramsListWithAnswers

def separateAnswers(samples):
    sample = []
    answers = []
    for i in range (len(samples)):
        sample.append(samples[i][0])
        answers.append(samples[i][1])
    return sample, answers

def connectAnswers(samples, answers):
    samplesWithAnswers = []
    if len(samples) != len(answers):
        raise ValueError('Cannot connect samples and answers! Different length of lists.')
    for i in range(len(samples)):
        samplesWithAnswers.append((samples[i],answers[i]))
    return samplesWithAnswers

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

#Return simple list of descriptors.
#Param: samples - array of tuples(image_descriptors, answer). image_descriptors.shape = (IMAGE_CELLS_COUNT, n_descriptors)
def singleLineDescriptors(samples):
    singleLine = []
    for index_image in range(len(samples)):
        for descList in samples[index_image][0]:
            singleLine += descList
    return singleLine

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

#Write to log file. File should be opened in global space!!!
def logWrite(string):
    global log
    log.write(string)
    log.flush()

#################################################
################# Constants #####################
#################################################
IMAGE_MIN_SIZE = 700
MIN_IMAGE_GRID_SIZE = 1
MAX_IMAGE_GRID_SIZE = 4
MIN_CLUSTER_COUNT_POWER = 3 
MAX_CLUSTER_COUNT_POWER = 8
CACHE_FILE_SEPARATION_COUNT = 10

#################################################
############## Global variables #################
#################################################
#Class = enum(A4 = 'A4', CARD = 'Business card', DUAL = 'Dual page', ROOT = 'Book list with root', SINGLE = 'Single book list', CHECK = 'Cash voucher(check)')
Class = enum(A4 = 0, CARD = 1, DUAL = 2, ROOT = 3, SINGLE = 4, CHECK = 5)

ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\Test\\'
#ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\'
#ROOT_Dir = 'D:\\ABBYY\\Abbyy photo\\Test0\\'
Dir_A4 = 'A4'
Dir_Card = 'Card'
Dir_Check = 'Check'
Dir_Dual = 'Dual'
Dir_Root = 'Root'
Dir_Single = 'Single'

CACHE_FILE_Descriptors = 'descriptors.bin'
CACHE_FILE_Test_Descriptors = 'test_descriptors.bin'
CACHE_FILE_Clusters = 'clusters.bin'
CACHE_FILE_Classifier = 'classifier.bin'
CACHE_FILE_Answers = 'answers.bin'
#CACHE_FILE_Samples = 'samples.bin'
CacheFile_Samples = 'samples.bin'
LogFile = 'log.txt'
StatisticsFile = 'stat.csv'




#################################################
############# Main functionality ################
#################################################
os.chdir(ROOT_Dir)
log = open(LogFile, 'w')


#TODO: REFACTOR CODE FROM BEGINING

#Building file list of train and test samples
sys.stdout.write('Looking for samples cache.\r')
if cacheExists(CacheFile_Samples):
    sys.stdout.write('Cache of samples is found.         \n')
    logWrite('Cache of samples is found.\n')
    trainSamples, testSamples, trainAnswers, testAnswers = loadFromCahe(CacheFile_Samples)
else:
    sys.stdout.write('Generating samples list.     \n')
    samplesFiles = loadDir(Dir_A4) + loadDir(Dir_Card) + loadDir(Dir_Check) + loadDir(Dir_Dual) + loadDir(Dir_Root) + loadDir(Dir_Single)
    logWrite('Generated files list. Total ' + str(len(samplesFiles)) + ' files.\n')
    answers = buildAnswers(samplesFiles)
    trainSamples, testSamples, trainAnswers, testAnswers = sklearn.cross_validation.train_test_split(samplesFiles,answers)
    logWrite('Train samples: ' + str(len(trainSamples)) + ' files.\n')
    logWrite('Test samples: ' + str(len(testSamples)) + ' files.\n')
    sys.stdout.write('Saving samples to cache file ' + CacheFile_Samples + '\r')
    data = trainSamples, testSamples, trainAnswers, testAnswers
    saveToCache(data, CacheFile_Samples)
    del data



for gridSize in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1):
    image_cells_count = gridSize**2
    CacheFile_Descriptors = 'CELL' + str(image_cells_count) + '_' + CACHE_FILE_Descriptors
    CacheFile_Test_Descriptors = 'CELL' + str(image_cells_count) + '_' + CACHE_FILE_Test_Descriptors
    CacheFile_Clusters = 'CELL' + str(image_cells_count) + '_' + CACHE_FILE_Clusters
    CacheFile_Classifier = 'CELL' + str(image_cells_count) + '_' + CACHE_FILE_Classifier
    CacheFile_Answers = 'CELL' + str(image_cells_count) + '_' + CACHE_FILE_Answers
    #CacheFile_Samples = 'CELL' + str(image_cells_count) + '_' + CACHE_FILE_Samples
    sys.stdout.write('IMAGE GRID SIZE = '+ str(gridSize) + 'X' + str(gridSize) + '. Param "image_cells_count" = ' + str(image_cells_count) + '\n')
    #Building descriptors for train samples
    #TODO: collect statistics
    sys.stdout.write('Looking for train descriptors cache.\r')
    if cacheExists(CacheFile_Descriptors + str(CACHE_FILE_SEPARATION_COUNT)):
        sys.stdout.write('Cache for descriptors is found.\n')
        logWrite('Cache for descriptors is found.\n')
    else:
        sys.stdout.write('Generating image descriptors .\n')
        partLength = int (numpy.ceil(floor(len(trainSamples)) / CACHE_FILE_SEPARATION_COUNT))                                               #Calculating number of files in one part
        for i in range(CACHE_FILE_SEPARATION_COUNT):
            sys.stdout.write('Part ' + str(i+1) + ' from ' + str(CACHE_FILE_SEPARATION_COUNT) +'.\n')
            logWrite('Generating image descriptors. Part ' + str(i+1) + ' from ' + str(CACHE_FILE_SEPARATION_COUNT) +'.\n')
            samplesFilesPart = trainSamples[i*partLength:(i+1)*partLength]                                                                  #Separating files to CACHE_FILE_SEPARATION_COUNT parts
            samplesKeyPoints, samplesDescriptors, samplesImageSizes = buildDescriptors(samplesFilesPart)                                    #Building descriptors and keypoints
            samplesSeparatedDescriptors = separateDescriptors(samplesKeyPoints, samplesDescriptors, samplesImageSizes, image_cells_count)   #Separating images to different cells count
            del samplesKeyPoints, samplesDescriptors, samplesImageSizes
            samplesSeparatedDescriptorsWithAnswers = connectAnswers(samplesSeparatedDescriptors, trainAnswers[i*partLength:(i+1)*partLength])    #Connecting samples with answers. It should help exclude samples when needed.
            sys.stdout.write('Saving image descriptors to cache ' + CacheFile_Descriptors + str(i+1) + '\n')
            logWrite('Saving image descriptors to cache ' + CacheFile_Descriptors + str(i+1) + '\n')
            saveToCache(samplesSeparatedDescriptorsWithAnswers,CacheFile_Descriptors + str(i+1))                                            #Saving data to cache
            del samplesSeparatedDescriptors, samplesSeparatedDescriptorsWithAnswers

    #Building descriptors for test samples
    #TODO: collect statistics
    sys.stdout.write('Looking for test descriptors cache.\r')
    if cacheExists(CacheFile_Test_Descriptors + str(CACHE_FILE_SEPARATION_COUNT)):
        sys.stdout.write('Cache for test descriptors is found.\n')
        logWrite('Cache for test descriptors is found.\n')
    else:
        sys.stdout.write('Generating test image descriptors .\n')
        partLength = int (numpy.ceil(floor(len(trainSamples)) / CACHE_FILE_SEPARATION_COUNT))                                               #Calculating number of files in one part
        for i in range(CACHE_FILE_SEPARATION_COUNT):
            sys.stdout.write('Part ' + str(i+1) + ' from ' + str(CACHE_FILE_SEPARATION_COUNT) +'.\n')
            logWrite('Generating test image descriptors. Part ' + str(i+1) + ' from ' + str(CACHE_FILE_SEPARATION_COUNT) +'.\n')
            samplesFilesPart = testSamples[i*partLength:(i+1)*partLength]                                                                  #Separating files to CACHE_FILE_SEPARATION_COUNT parts
            samplesKeyPoints, samplesDescriptors, samplesImageSizes = buildDescriptors(samplesFilesPart)                                    #Building descriptors and keypoints
            samplesSeparatedDescriptors = separateDescriptors(samplesKeyPoints, samplesDescriptors, samplesImageSizes, image_cells_count)   #Separating images to different cells count
            del samplesKeyPoints, samplesDescriptors, samplesImageSizes
            samplesSeparatedDescriptorsWithAnswers = connectAnswers(samplesSeparatedDescriptors, testAnswers[i*partLength:(i+1)*partLength])    #Connecting samples with answers. It should help exclude samples when needed.
            sys.stdout.write('Saving test image descriptors to cache ' + CacheFile_Test_Descriptors + str(i+1) + '\n')
            logWrite('Saving test image descriptors to cache ' + CacheFile_Test_Descriptors + str(i+1) + '\n')
            saveToCache(samplesSeparatedDescriptorsWithAnswers,CacheFile_Test_Descriptors + str(i+1))                                            #Saving data to cache
            del samplesSeparatedDescriptors, samplesSeparatedDescriptorsWithAnswers



    #Rebuild descriptors in one list and run partitional fit
    for power in range(MIN_CLUSTER_COUNT_POWER,MAX_CLUSTER_COUNT_POWER):
        n_clusters = 2**power
        #Rebuilding descriptors
        kmeans = MiniBatchKMeans(n_clusters = n_clusters,verbose = False)
        sys.stdout.write('Calculating cluster centers (' + str(n_clusters) + ' clusters).\n')
        for index_cacheFile in range(CACHE_FILE_SEPARATION_COUNT):
            sys.stdout.write('Loading cache ' + CacheFile_Descriptors + str(index_cacheFile+1) + '\r')
            descWithAnswers = loadFromCahe(CacheFile_Descriptors + str(index_cacheFile+1))
            simpleDesc = singleLineDescriptors(descWithAnswers)
            if len(simpleDesc) == 0:
                continue
            del descWithAnswers
            sys.stdout.write('Fitting kmeans.\r')
            kmeans.partial_fit(simpleDesc)
            del simpleDesc
    
        #Building histograms of descriptors distribution
        samplesHistogram = []
        for index_cacheFile in range(CACHE_FILE_SEPARATION_COUNT):
            sys.stdout.write('Loading cache ' + CacheFile_Descriptors + str(index_cacheFile+1) + '\r')
            descWithAnswers = loadFromCahe(CacheFile_Descriptors + str(index_cacheFile+1))
        
            sys.stdout.write('Calculating part ' + str(index_cacheFile+1) + ' from '+ str(CACHE_FILE_SEPARATION_COUNT) +'.\n')
            samplesHistogramPart = clasterizeInCellsWithAnswres(descWithAnswers, image_cells_count, kmeans,stat = True)
            samplesHistogram+=samplesHistogramPart
            del descWithAnswers
        sys.stdout.write('Saving clasterization histograms to cache.\n')
        saveToCache(samplesHistogram,CacheFile_Clusters + str(n_clusters))
        logWrite('Clasterization histograms cached to file ' + CacheFile_Clusters + str(n_clusters) + '.\n')

        #TRAIN CLASSIFIERS
        if cacheExists(CacheFile_Classifier + str(n_clusters)):
            sys.stdout.write('Loading classifier from cache.\n')
            l_svm, svm, rf = loadFromCahe(CacheFile_Classifier + str(n_clusters))
        else:
            sys.stdout.write('Training classifier.\n')
            logWrite('Started training classifier.\n')
            l_svm = sklearn.svm.LinearSVC()                         #Creating classifier object
            svm = sklearn.svm.SVC()
            rf = RandomForestClassifier()
            trainSam, trainAns = separateAnswers(samplesHistogram)
            l_svm.fit(trainSam, trainAns)                           #training classifier
            svm.fit(trainSam, trainAns)
            rf.fit(trainSam, trainAns)
            del samplesHistogram
            sys.stdout.write('Saving classifier to cache.\n')
            data = l_svm, svm, rf
            saveToCache(data, CacheFile_Classifier + str(n_clusters))
            logWrite('Classifier cached to file ' + CacheFile_Classifier + str(n_clusters) + '.\n')
    
        #TODO: use different classifiers

        #CHECK ACCURACY
        #Prepearing test data
        sys.stdout.write('Testing accuracy of classifier.\n')
        test_samplesHistogram = []
        for index_cacheFile in range(CACHE_FILE_SEPARATION_COUNT):
            sys.stdout.write('Loading cache ' + CacheFile_Test_Descriptors + str(index_cacheFile+1) + '\r')
            descWithAnswers = loadFromCahe(CacheFile_Test_Descriptors + str(index_cacheFile+1))
        
            sys.stdout.write('Calculating part ' + str(index_cacheFile+1) + ' from '+ str(CACHE_FILE_SEPARATION_COUNT) +'.\n')
            samplesHistogramPart = clasterizeInCellsWithAnswres(descWithAnswers, image_cells_count, kmeans,stat = False)
            test_samplesHistogram += samplesHistogramPart
            del descWithAnswers
    
        trainSam, trainAns = separateAnswers(test_samplesHistogram)
        accuracy_L = l_svm.score(trainSam, trainAns)
        accuracy_S = svm.score(trainSam, trainAns)
        accuracy_R = rf.score(trainSam, trainAns)

        logWrite('RESULTS OF TESTING OF CLUSSIFIER (CLUSTERS NUNBER = ' + str(n_clusters) + ' IMAGE CELLS NUMBER ' + str(image_cells_count) +'):\n')
        logWrite('Total train files: ' + str(len(trainSamples)) + '\n')
        logWrite('Total test files: ' + str(len(testSamples)) + '\n')
        logWrite('Accuracy of LINEAR SVM:' + str(accuracy_L) + ' %.\n')
        logWrite('Accuracy of SVM:' + str(accuracy_S) + ' %.\n')
        logWrite('Accuracy of RANDOM FOREST:' + str(accuracy_R) + ' %.\n')
    
        sys.stdout.write('RESULTS OF TESTING OF CLUSSIFIER (CLUSTERS NUNBER = ' + str(n_clusters) + ' IMAGE CELLS NUMBER ' + str(image_cells_count) +'):\n')
        sys.stdout.write('Total train files: ' + str(len(trainSamples)) + '\n')
        sys.stdout.write('Total test files: ' + str(len(testSamples)) + '\n')
        sys.stdout.write('Accuracy of LINEAR SVM:' + str(accuracy_L) + ' %.\n')
        sys.stdout.write('Accuracy of SVM:' + str(accuracy_S) + ' %.\n')
        sys.stdout.write('Accuracy of RANDOM FOREST:' + str(accuracy_R) + ' %.\n')
log.close()