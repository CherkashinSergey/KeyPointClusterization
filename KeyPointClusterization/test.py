import sklearn
import sklearn.grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

testSample = ((0,0,0), (1,1,1),(2,2,2))
trainSample = [(0,0,0), (1,1,1),(2,2,2)]
testAnswer = (0,1,2)
trainAnswer = [0,1,2]



param_grid = {'C':[0.1, 1, 10]}
clf = GridSearchCV(LinearSVC(), param_grid)
clf.fit(trainSample, trainAnswer)
print(clf.best_params_)