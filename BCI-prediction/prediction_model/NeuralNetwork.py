import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


class NeuralNetwork:
    def __init__(self):
        self.models = Sequential(
            [
                Dense(256, activation='relu', name='l1'),
                Dense(128, activation='relu', name='l2'),
                Dense(3, activation='linear', name='l3')
            ]
        )
        self.models.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(0.0005)
        )

    def fit(self, X_train, Y_train):
        self.models.fit(
            X_train, Y_train,
            epochs=100
        )

    def predict(self, Y_test):
        yhat = self.models.predict(Y_test)
        prediction = np.zeros(len(yhat))
        for i in range(len(yhat)):
            prediction[i] = np.argmax(yhat[i])
        return prediction
