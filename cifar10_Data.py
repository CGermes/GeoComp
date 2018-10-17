# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:58:41 2018

@author: carlos
"""

from __future__ import division
from keras.datasets import cifar10
import matplotlib.pylab as plt
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import utils
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions
img_x, img_y = 32, 32

x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
x_test = x_test / 255
y_test = y_test.flatten()

# load json and create model
json_file = open('model_cifar.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


algoCNN = model_from_json(loaded_model_json)
# load weights into new model
algoCNN.load_weights("model_cifar.h5")
algoCNN.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print "============ RESULTADOS CNN ============="
y_predicted = np.argmax(algoCNN.predict(x_test), axis=1)
print "Validation: ", round(100*sum(y_predicted==y_test)/x_test.shape[0], 2), "%"
confusionMatrix =  confusion_matrix(y_test, y_predicted)
utils.plot_confusion_matrix(confusionMatrix, classes=range(10), normalize=False, title='Convolutional Neural Network Confusion matrix')
plt.show()  

N=x_test.shape[0]

labels = ["avion", "coche", "pajaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camion"]

def test():
    plt.figure()
    i = int(np.random.uniform(0, N))
    plt.imshow(x_test[i])
    print "Es un ", labels[y_predicted[i]]
