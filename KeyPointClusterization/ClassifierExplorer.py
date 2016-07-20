import os
import pickle
import sklearn
import numpy

#Gets data from binary file "fileName"
def loadFromCahe(fileName):
    cache = open(fileName, 'rb')
    #data = zlib.decompress(data)
    data = pickle.load(cache)
    cache.close()
    return data

ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\Results\\Classifiers'
Classifier_Dump = 'GRID_5-6CL7-9LinearSVM.bin'
#hist_file = 'hist.csv'

MIN_IMAGE_GRID_SIZE = 5
MAX_IMAGE_GRID_SIZE = 6
MIN_CLUSTER_COUNT_POWER = 7
MAX_CLUSTER_COUNT_POWER = 9
HIST_LENGTH = 200
CLASS_NAMES = ('A4', 'CARD', 'DUAL', 'ROOT', 'SINGLE', 'CHECK')

os.chdir(ROOT_Dir)
#csv = open(hist_file, 'w')
LinearSVM_List = loadFromCahe(Classifier_Dump)



for gridSize in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1):
    n_cells = gridSize**2
    for power in range(MIN_CLUSTER_COUNT_POWER,MAX_CLUSTER_COUNT_POWER+1):
            n_clusters = 2**power
            l_svm =LinearSVM_List[gridSize-MIN_IMAGE_GRID_SIZE][power - MIN_CLUSTER_COUNT_POWER]
            current_CSV_name = 'CL_' + str(n_clusters) + '_GRID_' + str(gridSize) + 'X' + str(gridSize) + 'hist.csv'
            csv = open(current_CSV_name, 'w')
            Max = 0
            Min = 0
            for index_class in range(len(l_svm.coef_)):
            #Initialize sizes
                c_max = max(l_svm.coef_[index_class])
                c_min = min(l_svm.coef_[index_class])
                Max = max(c_max, Max)
                Min = min(c_min, Min)
            Max += 0.1
            Min -= 0.1
            bucket_len = float(Max - Min)/HIST_LENGTH
            #Write header in csv
            csv.write(';')
            for i in range(HIST_LENGTH):
                csv.write(str(Min + i*bucket_len).replace('.',',') + ';')
            csv.write('\n')

            for index_class in range(len(l_svm.coef_)):
                #Make histogram                
                hist = [0 for x in range(HIST_LENGTH)]
                for value in l_svm.coef_[index_class]:
                    index_bucket = int(numpy.floor(float(value - Min)/bucket_len))
                    hist[index_bucket] +=1
                #Write histogram in csv
                csv.write(CLASS_NAMES[index_class] + ';')
                for bucket in hist:
                    csv.write(str(bucket) + ';')
                csv.write('\n')
                csv.flush()
            csv.close()

#power = 7
#gridSize = 1
#l_svm =LinearSVM_List[0][0]
#n_cells = gridSize**2
#n_clusters = 2**power

#bucket_len = float(Max - Min)/HIST_LENGTH

#for index_class in range(len(l_svm.coef_)):
#    hist = [0 for x in range(HIST_LENGTH)]
#    for value in l_svm.coef_[index_class]:
#        index_bucket = int(numpy.floor(float(value - Min)/bucket_len))
#        hist[index_bucket] +=1

#    for bucket in hist:
#        csv.write(str(bucket) + ';')
#    csv.write('\n')
#    csv.flush()
#csv.close()
