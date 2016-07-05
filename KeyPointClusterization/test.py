MIN_IMAGE_GRID_SIZE = 2
MAX_IMAGE_GRID_SIZE = 4
LinearSVM = [list() for x in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1)]
SVM = [list() for x in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1)]
RandomForest = [list() for x in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1)]

for gridSize in range(MIN_IMAGE_GRID_SIZE,MAX_IMAGE_GRID_SIZE+1):
    SVM[gridSize-MIN_IMAGE_GRID_SIZE].append(1)