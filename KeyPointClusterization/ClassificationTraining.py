import os
import sys
#import shutil
import pickle
import cv2
import sklearn
import numpy
from sklearn.cluster import *
from sklearn.ensemble import *
#from pylab import *
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
    
    global DocType
    answers = []
    for file in fileList:
        if os.path.dirname(file) == dir_A4:
            answers.append(DocType.A4)
        elif os.path.dirname(file) == dir_Card:
            answers.append(DocType.CARD)
        elif os.path.dirname(file) == dir_Check:
            answers.append( DocType.CHECK)
        elif os.path.dirname(file) == dir_Dual:
            answers.append(DocType.DUAL)
        elif os.path.dirname(file) == dir_Root:
            answers.append(DocType.ROOT)
        elif os.path.dirname(file) == dir_Single:
            answers.append(DocType.SINGLE)
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

    global DocType
    answers = singleAnswerList(dir_A4, DocType.A4)
    answers += singleAnswerList(dir_Card, DocType.CARD)
    answers += singleAnswerList(dir_Check, DocType.CHECK)
    answers += singleAnswerList(dir_Dual, DocType.DUAL)
    answers += singleAnswerList(dir_Root, DocType.ROOT)
    answers += singleAnswerList(dir_Single, DocType.SINGLE)
    
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
    global MAX_KEYPOINTS_PER_IMAGE
    global TotalKeyPointsCount
    global HESSIAN_THRESHOLD
    keyPoints = []
    descriptors = []
    imageSizes = []
    sift = cv2.SIFT(nfeatures = MAX_KEYPOINTS_PER_IMAGE)
    #sift = cv2.SIFT()
    #surf = cv2.SURF(hessianThreshold = HESSIAN_THRESHOLD)
    #surf = cv2.SURF()
    
    for i in range (len(sampleFileList)):
        sys.stdout.write('Building descriptors for image ' + str(i+1) + ' of ' + str(len(sampleFileList)) + ' (' + (os.path.split(os.path.dirname(sampleFileList[i])))[1] +')...\r')
        file = sampleFileList[i]
        #logWrite('Buildind descriptors for image ' + file + '\n')
        #Building keypionts, descriptors,
        img = cv2.imread(file)              #read image
        img = fitImage(img)                 #resize image
        imageSizes.append((img.shape[0],img.shape[1]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray,None) #build keypoints and descriptors on gray image
        #kp, des = sift.compute(gray, kp)
        TotalKeyPointsCount += len(kp)
        if des is None:
            print('Cannot build descriptors to file ' + file)
            kp = []
            des = []
        #else:
        #    descriptors.append(des)
        #    keyPoints.append(kp)
        descriptors.append(des)
        keyPoints.append(kp)
    sys.stdout.write('Building descriptors complete.                              \n')
    return keyPoints, descriptors, imageSizes

#Clasterizes array "samples [samplesDescriptors]" to n clasters
def clasterizeInCells(samples, n_imageCells, kmeans, stat = False):
    if stat:
        global StatisticsFile
        statFileName = 'CL_' + str(kmeans.n_clusters) + 'CELL_' + str(n_imageCells) + StatisticsFile
        statFile = open(statFileName,'a')
        CSVSeparator = ';'
    histogramsList = [] # shape (n_images, n_clusters*n_imageCells)
    excludedSamplesCount = 0
    for index_Image in range(len(samples)):
        sys.stdout.write('Clasterizing descriptors for image ' + str(index_Image+1) + ' from ' + str(len(samples)) + '...\r')
        imageHist = []
        for index_Cell in range(n_imageCells):
            if len(samples[index_Image][index_Cell]) == 0:
                hist = [0 for x in range(n_clusters)]                                   #in case when some image cell has no descriptors
            else:
                hist = kmeans.predict(samples[index_Image][index_Cell])
                hist = buildHistogram(hist, n_clusters)
            if stat:                                                                    #writing statistics
                for value in hist:
                    statFile.write(str(value) + CSVSeparator)
            #hist = normalizeHistogram(hist)                                             #Should try normalization on whole imageHist
            imageHist += hist
        imageHist = normalizeHistogram(imageHist)
        histogramsList.append(imageHist)
        if stat:                                                                        #Should be refactored
                statFile.write('\n')
                statFile.flush()
    if stat: statFile.close()
    return histogramsList


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
#def separateDescriptors(keyPoints, descriptors, imageSize, n_cells):
#    new_descriptors = [[list() for x in range(n_cells)] for i in range(len(keyPoints))]
#    side = int(numpy.sqrt(n_cells))
#    for image_index in range(len(keyPoints)):
#       for kp_index in range(len(keyPoints[image_index])):
#            cell_X = int(numpy.floor(keyPoints[image_index][kp_index].pt[0]/(imageSize[image_index][1]/side)))
#            cell_Y = int(numpy.floor(keyPoints[image_index][kp_index].pt[1]/(imageSize[image_index][0]/side)))
#            descriptor_new_index = cell_X*side +cell_Y
#            (new_descriptors[image_index][descriptor_new_index]).append(descriptors[image_index][kp_index])
#    return new_descriptors

def separateDescriptors(keyPoints, descriptors, imageSize, n_cells):
    new_descriptors = [[list() for x in range(n_cells)] for y in range(len(keyPoints))]
    side = int(numpy.sqrt(n_cells))
    for image_index in range(len(keyPoints)):
       for kp_index in range(len(keyPoints[image_index])):
            cell_X = int(numpy.floor(keyPoints[image_index][kp_index][0]/(float(imageSize[image_index][1])/side)))
            cell_Y = int(numpy.floor(keyPoints[image_index][kp_index][1]/(float(imageSize[image_index][0])/side)))
            descriptor_new_index = cell_X*side +cell_Y
            (new_descriptors[image_index][descriptor_new_index]).append(descriptors[image_index][kp_index])
    return new_descriptors

#Return simple list of descriptors.
#Param: samples - array of tuples(image_descriptors, answer). image_descriptors.shape = (IMAGE_CELLS_COUNT, n_descriptors)
def singleLineDescriptorsWithAnswers(samples):
    singleLine = []
    for index_image in range(len(samples)):
        for descList in samples[index_image][0]:
            singleLine += descList
    return singleLine

def singleLineDescriptors(samples):
    singleLine = []
    for image in samples:
        for descriptor in image:
            singleLine.append(descriptor)
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

def transformKP(keypoints):
    serial = []
    for image in keypoints:
        temp_image = []
        for kp in image:
            temp_image.append(kp.pt)
        serial.append(temp_image)
    return serial

#Displays information in convinient view in console
def display(info = None, init = False, step = None):
    #clear()
    os.system('cls' if os.name == 'nt' else 'clear')
    display.n_cluster = 8
    display.n_imageCell = 1
    display.step = 1
    display.iteration = 0
    display.stepCount = 5

    sys.stdout.write('Classifier training programm.\n')
    sys.stdout.write('Iteration '+ str(display.iteration) + '.\n')
    sys.stdout.write('Parameters: cluster count = ' + str(display.n_cluster) + '; image cells count = ' + str(display.n_imageCell) + '.\n')
    sys.stdout.write('Step '+ str(display.step) +'/' + str(display.stepCount) +'.\n')
    if info is not None: 
        sys.stdout.write(info + '\n')
    display.iteration += 1

#Write to log file. File should be opened in global space!!!
def logWrite(string):
    global log
    log.write(string)
    log.flush()

#Returns accuracy and saves distribution of answers in csv file "fileNamePrefix + 'accuracy.csv'"
def checkAccuracyAndLog(cl_answers, true_answers, fileNamePrefix):
    def getIndex(answer):
        global DocType
        if   answer == DocType.A4    : return 0
        elif answer == DocType.CARD  : return 1
        elif answer == DocType.DUAL  : return 2
        elif answer == DocType.ROOT  : return 3
        elif answer == DocType.SINGLE: return 4
        elif answer == DocType.CHECK : return 5
        else: return -1
    separator = ';'
    csvName = fileNamePrefix + 'accuracy.csv'
    csv = open(csvName, 'w')
    csv.write('A4; CARD; DUAL; ROOT; SINGLE; CHECK;\n')            #write headline
    n_CorrectAnswers = 0
    accuracy_matrix = [[0 for y in range(6)] for x in range(6)] #creating table of counters
    for index_answer in range(len(cl_answers)):
        x_cell = getIndex(true_answers[index_answer])
        y_cell = getIndex(cl_answers[index_answer])
        accuracy_matrix[x_cell][y_cell] += 1
        if cl_answers[index_answer] == true_answers[index_answer]: n_CorrectAnswers += 1
    for row in accuracy_matrix:
        for value in row:
            csv.write(str(value) + separator)
        csv.write('\n')
    csv.flush()
    csv.close()
    return float(n_CorrectAnswers)/len(cl_answers)

#################################################
################# Constants #####################
#################################################
IMAGE_MIN_SIZE = 700
MIN_IMAGE_GRID_SIZE = 1
MAX_IMAGE_GRID_SIZE = 6
MIN_CLUSTER_COUNT_POWER = 7
MAX_CLUSTER_COUNT_POWER = 10
CACHE_FILE_SEPARATION_COUNT = 1
PARTIAL_FIT_COUNT = 10
TRAIN_SIZE = 0.5
MAX_KEYPOINTS_PER_IMAGE = 2000
HESSIAN_THRESHOLD = 600

#################################################
############## Global variables #################
#################################################
#Class = enum(A4 = 'A4', CARD = 'Business card', DUAL = 'Dual page', ROOT = 'Book list with root', SINGLE = 'Single book list', CHECK = 'Cash voucher(check)')
DocType = enum(A4 = 0, CARD = 1, DUAL = 2, ROOT = 3, SINGLE = 4, CHECK = 5)

#ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\Test\\'
ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\'
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
TotalKeyPointsCount = 0




#################################################
############# Main functionality ################
#################################################
os.chdir(ROOT_Dir)
log = open(LogFile, 'w')
sys.stdout.softspace = True
if cacheExists(CACHE_FILE_Descriptors):
    sys.stdout.write('loading cache.\n')
    trainSamples, testSamples, trainAnswers, testAnswers = loadFromCahe(CACHE_FILE_Answers)
    samplesKeyPoints, samplesDescriptors, samplesImageSizes = loadFromCahe(CACHE_FILE_Descriptors)
    sys.stdout.write('Total train files: ' + str(len(trainSamples)) + '\n')
    sys.stdout.write('Total test files: ' + str(len(testSamples)) + '\n')
    logWrite('Train samples: ' + str(len(trainSamples)) + ' files.\n')
    logWrite('Test samples: ' + str(len(testSamples)) + ' files.\n')
else:
    #Generating file lists
    sys.stdout.write('Generating samples list.\n')
    samplesFiles = loadDir(Dir_A4) + loadDir(Dir_Card) + loadDir(Dir_Check) + loadDir(Dir_Dual) + loadDir(Dir_Root) + loadDir(Dir_Single)
    logWrite('Generated files list. Total ' + str(len(samplesFiles)) + ' files.\n')
    answers = buildAnswers(samplesFiles)
    trainSamples, testSamples, trainAnswers, testAnswers = sklearn.cross_validation.train_test_split(samplesFiles,answers, train_size = TRAIN_SIZE)
    sys.stdout.write('Total train files: ' + str(len(trainSamples)) + '\n')
    sys.stdout.write('Total test files: ' + str(len(testSamples)) + '\n')
    logWrite('Train samples: ' + str(len(trainSamples)) + ' files.\n')
    logWrite('Test samples: ' + str(len(testSamples)) + ' files.\n')
    del samplesFiles, answers
    

    #Building train samples
    sys.stdout.write('Generating image descriptors .\n')
    logWrite('Generating image descriptors.\n')
    samplesKeyPoints, samplesDescriptors, samplesImageSizes = buildDescriptors(trainSamples)                                    #Building descriptors and keypoints
    samplesKeyPoints = transformKP(samplesKeyPoints)
    sys.stdout.write('Total train keypoints found: ' + str(TotalKeyPointsCount) +'\n')
    logWrite('Total train keypoints found: ' + str(TotalKeyPointsCount) +'\n')
    TotalKeyPointsCount = 0

    sys.stdout.write('Saving cache.\n')
    data = trainSamples, testSamples, trainAnswers, testAnswers
    saveToCache(data, CACHE_FILE_Answers)
    data = samplesKeyPoints, samplesDescriptors, samplesImageSizes
    saveToCache(data, CACHE_FILE_Descriptors)

#Clasterizing and training
LinearSVM = [list() for x in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1)]
Kmeans = [list() for x in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1)]
ClussifierDumpName = 'GRID_'+ str(MIN_IMAGE_GRID_SIZE) + '-' + str(MAX_IMAGE_GRID_SIZE) + 'CL' + str(MIN_CLUSTER_COUNT_POWER) + '-' + str(MAX_CLUSTER_COUNT_POWER) + CACHE_FILE_Classifier
if cacheExists(ClussifierDumpName):
    del samplesKeyPoints, samplesDescriptors, samplesImageSizes
    LinearSVM, Kmeans = loadFromCahe(ClussifierDumpName)
else:
#TRAINING CLASSIFIERS
    for gridSize in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1):
        image_cells_count = gridSize**2
        #Separate descriptors on image cells
        sys.stdout.write('IMAGE GRID SIZE = '+ str(gridSize) + 'X' + str(gridSize) + '. Param "image_cells_count" = ' + str(image_cells_count) + '\n')
        #Rebuild descriptors in one list and run partitional fit
        for power in range(MIN_CLUSTER_COUNT_POWER,MAX_CLUSTER_COUNT_POWER+1):
            n_clusters = 2**power
            #Rebuilding descriptors
            sys.stdout.write('Calculating cluster centers (' + str(n_clusters) + ' clusters).\n')
            kmeans = MiniBatchKMeans(n_clusters = n_clusters,verbose = False)
            partLength = int (numpy.ceil(numpy.floor(len(samplesKeyPoints)) / PARTIAL_FIT_COUNT))
            for index_part in range(PARTIAL_FIT_COUNT):
                sys.stdout.write('Part ' + str(index_part+1) + '/' + str(PARTIAL_FIT_COUNT) +'. Separating descriptors.\r')
                if len(samplesDescriptors[index_part*partLength:(index_part+1)*partLength]) == 0 : continue                      #don't do anything if part of sample is empty
                simpleDesc = singleLineDescriptors(samplesDescriptors[index_part*partLength:(index_part+1)*partLength])
                sys.stdout.write('Part ' + str(index_part+1) + '/' + str(PARTIAL_FIT_COUNT) +'. Fitting kmeans.        \r')
                #if (len(simpleDesc) <= n_clusters): continue
                kmeans.partial_fit(simpleDesc)
                del simpleDesc
        
            samplesSeparatedDescriptors = separateDescriptors(samplesKeyPoints,samplesDescriptors,samplesImageSizes,image_cells_count)                                          
            #Building histograms of descriptors distribution
            samplesHistogram = clasterizeInCells(samplesSeparatedDescriptors, image_cells_count, kmeans,stat = True)
            del samplesSeparatedDescriptors
            logWrite('Clasterization histograms constructed (' + str(n_clusters) + ' clusters).\n')


            #trainSam, trainAns = separateAnswers(samplesHistogram)
            trainSam = samplesHistogram
            trainAns = trainAnswers
            del samplesHistogram
            #training classifiers
            sys.stdout.write('Training classifier.\t\t\t\t\t\t\n')
            logWrite('Started training classifier.\n')
            l_svm = sklearn.svm.LinearSVC()                         #Creating classifier object
            
            l_svm.fit(trainSam, trainAns)                           #training classifier
            
            LinearSVM[gridSize-MIN_IMAGE_GRID_SIZE].append(l_svm)
            Kmeans[gridSize-MIN_IMAGE_GRID_SIZE].append(kmeans)

    del samplesKeyPoints, samplesDescriptors, samplesImageSizes
    saveToCache(LinearSVM, 'GRID_'+ str(MIN_IMAGE_GRID_SIZE) + '-' + str(MAX_IMAGE_GRID_SIZE) + 'CL' + str(MIN_CLUSTER_COUNT_POWER) + '-' + str(MAX_CLUSTER_COUNT_POWER) + 'LinearSVM.bin')

if cacheExists(CACHE_FILE_Test_Descriptors):
    sys.stdout.write('loading test cache.\n')
    testKeyPoints, testDescriptors, testImageSizes = loadFromCahe(CACHE_FILE_Test_Descriptors)
else:
    #Building test samples
    sys.stdout.write('Generating test image descriptors .\n')
    logWrite('Generating test image descriptors.\n')
    testKeyPoints, testDescriptors, testImageSizes = buildDescriptors(testSamples)                                    #Building descriptors and keypoints
    testKeyPoints = transformKP(testKeyPoints)
    sys.stdout.write('Total test keypoints found: ' + str(TotalKeyPointsCount) +'\n')
    logWrite('Total test keypoints found: ' + str(TotalKeyPointsCount) +'\n')
    TotalKeyPointsCount = 0
    sys.stdout.write('Saving test descriptors to file.\r')
    data = testKeyPoints, testDescriptors, testImageSizes
    saveToCache(data, CACHE_FILE_Test_Descriptors)
    del data

sys.stdout.write('Checking accuracy.\n')
logWrite('Started accuracy checking.\n')
#Checking accuracy
for gridSize in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1):
    image_cells_count = gridSize**2
    #Separate descriptors on image cells
    sys.stdout.write('IMAGE GRID SIZE = '+ str(gridSize) + 'X' + str(gridSize) + '. Param "image_cells_count" = ' + str(image_cells_count) + '\n')
    for power in range(MIN_CLUSTER_COUNT_POWER,MAX_CLUSTER_COUNT_POWER+1):
        n_clusters = 2**power
        #Rebuilding descriptors
        kmeans = Kmeans[gridSize-MIN_IMAGE_GRID_SIZE][power - MIN_CLUSTER_COUNT_POWER]
        samplesSeparatedDescriptors = separateDescriptors(testKeyPoints, testDescriptors, testImageSizes, image_cells_count)   #Separating images to different cells count
        samplesSeparatedDescriptorsWithAnswers = connectAnswers(samplesSeparatedDescriptors, testAnswers)    #Connecting samples with answers. It should help exclude samples when needed.
        del samplesSeparatedDescriptors
    
        #Building histograms of descriptors distribution
        test_samplesHistogram = clasterizeInCellsWithAnswres(samplesSeparatedDescriptorsWithAnswers, image_cells_count, kmeans,stat = False)
        del samplesSeparatedDescriptorsWithAnswers
        logWrite('Clasterization histograms constructed on' + str(n_clusters) + '.\n')


        testSam, testAns = separateAnswers(test_samplesHistogram)
        del test_samplesHistogram
        #training classifiers
        l_svm = LinearSVM[gridSize-MIN_IMAGE_GRID_SIZE][power - MIN_CLUSTER_COUNT_POWER]                         #Creating classifier object

        accuracy_L = l_svm.score(testSam, testAns)
        #cl_answers = l_svm.predict(testSam)
        #accuracy_L = checkAccuracyAndLog(cl_answers,testAns,'CL_' + str(n_clusters) + 'CELL_' + str(image_cells_count))
        del testSam, testAns
        logWrite('RESULTS OF TESTING OF CLUSSIFIER (CLUSTERS NUNBER = ' + str(n_clusters) + ' IMAGE CELLS NUMBER ' + str(image_cells_count) +'):\n')
        logWrite('Accuracy of LINEAR SVM:' + str(accuracy_L) + ' %.\n')
    
        sys.stdout.write('RESULTS OF TESTING OF CLUSSIFIER (CLUSTERS NUNBER = ' + str(n_clusters) + ' IMAGE CELLS NUMBER ' + str(image_cells_count) +'):\n')
        sys.stdout.write('Accuracy of LINEAR SVM:' + str(accuracy_L) + ' %.\n')

log.close()