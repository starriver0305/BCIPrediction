import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self):
        self.knn = KNeighborsClassifier()

    def fit(self, X_train, Y_train):
        self.knn.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.knn.predict(X_test)
