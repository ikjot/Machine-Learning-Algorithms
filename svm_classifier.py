import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


from sklearn.base import BaseEstimator, ClassifierMixin
from final_fv import *
import scipy.io

from sklearn.utils import check_random_state


def makingProj(v, z=1):
    no_features = v.shape[0]
    #     temp_u = np.sort(v)[1::-1]
    # temp_u = np.sort(v)[:-2:]

    temp_u = np.sort(v)[::-1]
    temp_c = np.cumsum(temp_u) - z

    # i = np.arange(temp_u) + 1

    #print(i)
    # i = np.arange(100) + 1

    i = np.arange(no_features) + 1
    condition = temp_u - temp_c / i > 0
    r = i[condition][-1]

    # theta = temp_c - temp_c / i > 0
    #print(theta)

    theta = temp_c[condition][-1] / float(r)
    weight = np.maximum(v - theta, 0)
    return weight


class svm_classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1):
        self.C = C

    def _vl(self, gi, y, i):
        a = np.inf
        for temp in range(gi.shape[0]):
            # if self.dual_coef_[temp,i]  >=  2 :
                    # print("hello")
            if self.dual_coef_[temp,i]  >=  0 :
                continue
            a = min(a, gi[temp])

        return gi.max() - a

    def _optimize(self, der, X, n, i):
        temp = np.zeros(der.shape[0])
        temp[X[i]] = self.C



        aa = n[i] * (temp - self.dual_coef_[:, i]) + der / n[i]
        z = self.C * n[i]

        b = makingProj(aa, z)
        # print(b)
        return temp - self.dual_coef_[:, i] - b / n[i]


    def _derivative(self, X_arr, y, i):
        temp = np.dot(X_arr[i], self.coef_.T) + 1
        # print(temp)
        temp[y[i]] -= 1
        return temp


    def fit(self, X, y):
        num_of_samples, feat = X.shape
        # print(feat)
        # print(num_of_samples)
        self._encoder = LabelEncoder()


        y = self._encoder.fit_transform(y)

        no_of_cl = len(self._encoder.classes_)
        self.dual_coef_ = np.zeros((no_of_cl, num_of_samples), dtype=np.float64)

        #print(self.dual_coef_)
        self.coef_ = np.zeros((no_of_cl, feat))

        n_temp = np.sqrt(np.sum(X ** 2, axis=1))

        random = check_random_state(None)
        indexes = np.arange(num_of_samples)
        random.shuffle(indexes)
        vTemp = None
        for i_temp in range(50):
            sum_total = 0

            for z in range(num_of_samples):
                i = indexes[z]

                der = self._derivative(X, y, i)
                v = self._vl(der, y, i)
                sum_total += v
                diff = self._optimize(der, y, n_temp, i)

                self.coef_ += (diff * X[i][:, np.newaxis]).T
                self.dual_coef_[:, i] += diff

            if i_temp == 0:
                vTemp = sum_total
            r = sum_total / vTemp
            if r < 0.01:
                break

        return self

    def predict(self, X):
        d = np.dot(X, self.coef_.T)
        prediction = d.argmax(axis=1)
        return self._encoder.inverse_transform(prediction)


X_train,y_train,X_test = loaddataTrain()



clf = svm_classifier()
clf.fit(X_train[:,1:],y_train)
y_pred = clf.predict(X_test[:,1:])
for i in range(X_test.shape[0]):
    print (" {} : {}".format(X_test[i][0],y_pred[i]))
# print(clf.score(X_test,y_test))
