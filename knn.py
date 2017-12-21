import numpy as np
import scipy.io
import collections
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from final_fv import *


def knn(x_train,y_train,x_test):
	dist = np.array([[np.sqrt(np.sum(np.square(y-x))) for x in x_train] for y in x_test])

	k = 5
	idx = np.argsort(dist)
	j = 0
	correct_predict = 0
	preds = []
	for index in idx:
		arr = []
		for i in index[:k]:
			arr.append(y_train[i]) 
		most_common,num_most_common = collections.Counter(arr).most_common(1)[0]
		preds.append(most_common)
	return preds


X_train,y_train,X_test = loaddataTrain()
y_pred = knn(X_train[:,1:],y_train,X_test[:,1:])
for i in range(X_test.shape[0]):
    print (" {} : {}".format(X_test[i][0],y_pred[i]))
	





