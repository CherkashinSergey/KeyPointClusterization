import os
import pickle
import sklearn

#Gets data from binary file "fileName"
def loadFromCahe(fileName):
    cache = open(fileName, 'rb')
    #data = zlib.decompress(data)
    data = pickle.load(cache)
    cache.close()
    return data

ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\Results\\SIFT2000_128-256'
Classifier_Dump = 'GRID_5-6CL7-9LinearSVM.bin'
MIN_IMAGE_GRID_SIZE = 5
MAX_IMAGE_GRID_SIZE = 6
MIN_CLUSTER_COUNT_POWER = 7
MAX_CLUSTER_COUNT_POWER = 9
os.chdir(ROOT_Dir)

LinearSVM_List = loadFromCahe(Classifier_Dump)

for gridSize in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1):
    for power in range(MIN_CLUSTER_COUNT_POWER,MAX_CLUSTER_COUNT_POWER+1):
            n_clusters = 2**power
            l_svm =LinearSVM_List[gridSize-MIN_IMAGE_GRID_SIZE][power - MIN_CLUSTER_COUNT_POWER]
