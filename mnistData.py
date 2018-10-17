# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:26:46 2018

@author: carlos
"""
from __future__ import division
from Tkinter import *
from PIL import Image, ImageDraw
from algorithms import Perceptron, Least_Squares, Linear_Discriminant_Fisher, Multilayer_Perceptron
from algorithms import PCA, LDA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
import utils

class Paint(object):

    WIDTH = 560
    HEIGHT = 560
    DEFAULT_PEN_SIZE = 50
    DEFAULT_COLOR = 'black'
    FILE = "/home/carlos/Documentos/Optimizacion/tmp_canvas"
    
    def __init__(self):
        self.root = Tk()


        self.color_button = Button(self.root, text='done', command=self.done)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='erase', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=1, to=100, orient=HORIZONTAL)
        self.choose_size_button.set(self.DEFAULT_PEN_SIZE)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=self.WIDTH, height=self.HEIGHT)
        self.c.grid(row=1, columnspan=5)
        self.imageInMemory = Image.new("L", (self.WIDTH, self.HEIGHT), 'white')
        self.drawInMemory = ImageDraw.Draw(self.imageInMemory)
      
        # Cargamos el conjunto de datos
        mnist = fetch_mldata('MNIST original', data_home=".")
        fullDataset = np.array(mnist.data.T)
        fullDatasetClases = np.array(mnist.target.astype(int) +1)
        
        
        originalFullDataset = fullDataset
        # Preparamos el preprocesado
        self.preprocesor = []
        self.preprocesor.append(PCA(0.2))
        fullDataset = self.preprocesor[0].fit_transform(fullDataset)
        
#        self.preprocesor.append(LDA(0.05))
#        fullDataset = self.preprocesor[1].fit_transform(fullDataset, fullDatasetClases)
        
        
        # preparamos las particiones de validacion y entrenamiento haciendo una mezcla (los datos van ordenados por clases)
        N=fullDataset.shape[1]
        K = max(fullDatasetClases)
        
        permutation = np.array(range(N))
        np.random.shuffle(permutation)
        validationSize=int(0.2*N)
        fullDataset=fullDataset[:, permutation]
        originalFullDataset = originalFullDataset[:, permutation]
        fullDatasetClases = fullDatasetClases[permutation]
        
        # Cargamos las particiones haciendo el preprocesado
        self.data=fullDataset[:,validationSize:]
        self.clases= fullDatasetClases[validationSize:]
        self.validationData = fullDataset[:,:validationSize]
        self.validationClases = fullDatasetClases[:validationSize]
        
        D = self.data.shape[0]

        #DEBUG Vemos que todas las clases estan representadas. Aunque se verá en la matriz de confusion
#        unique, counts = np.unique(self.validationClases, return_counts=True) 
#        print "Elementos en Validacion: ", dict(zip(unique, counts))
        
        self.algo=Perceptron()
        print "Fitting with Perceptron"
        self.algo.fit(self.data, self.clases ) 
        print "======= RESULTADOS PERCEPTRON ========"
        print "Accuracy: ", round(100*sum(self.algo.predict(self.data)==self.clases)/self.data.shape[1], 2), "%"
        print "Validation: ", round(100*sum(self.algo.predict(self.validationData)==self.validationClases)/self.validationData.shape[1], 2), "%"
        confusionMatrix =  confusion_matrix(self.validationClases, self.algo.predict(self.validationData))
        
        utils.plot_confusion_matrix(confusionMatrix, classes=range(10), normalize=False, title='Perceptron Confusion matrix')
        plt.figure()
        plt.show()
        
        self.algoLeastSquare=Least_Squares()
        print "Fitting with Least Squares"
        self.algoLeastSquare.fit(self.data, self.clases )
        print "======= RESULTADOS LEAST SQUARES ========"
        print "Accuracy: ", round(100*sum(self.algoLeastSquare.predict(self.data)==self.clases)/self.data.shape[1], 2), "%"
        print "Validation: ", round(100*sum(self.algoLeastSquare.predict(self.validationData)==self.validationClases)/self.validationData.shape[1], 2), "%"
        confusionMatrix =  confusion_matrix(self.validationClases, self.algoLeastSquare.predict(self.validationData))
        utils.plot_confusion_matrix(confusionMatrix, classes=range(10), normalize=False, title='Minimos Cuadrados Confusion matrix')
        plt.figure()      
        
        self.algoLinearDiscriminantAnalysis=Linear_Discriminant_Fisher()
        print "Fitting with Fisher LDA"
        self.algoLinearDiscriminantAnalysis.fit(self.data, self.clases )
        print "======= RESULTADOS FISHER LDA ========"
        print "Accuracy:   ", round(100*sum(self.algoLinearDiscriminantAnalysis.predict(self.data)==self.clases)/self.data.shape[1], 2), "%"
        print "Validation: ", round(100*sum(self.algoLinearDiscriminantAnalysis.predict(self.validationData)==self.validationClases)/self.validationData.shape[1], 2), "%"
        confusionMatrix =  confusion_matrix(self.validationClases, self.algoLinearDiscriminantAnalysis.predict(self.validationData))
        utils.plot_confusion_matrix(confusionMatrix, classes=range(10), normalize=False, title='LDA Fisher Confusion matrix')
        plt.figure() 

        self.algoNeuralNetwork=Multilayer_Perceptron(D, [100], K, 'softmax')
        print "Fitting with Neural Network"
        self.algoNeuralNetwork.fit(self.data, self.clases, 0.3, 10000, 100)
        print "======= RESULTADOS NEURAL NETWORK ========"
        print "Accuracy:   ", round(100*sum(self.algoNeuralNetwork.predict(self.data)==self.clases)/self.data.shape[1], 2), "%"
        print "Validation: ", round(100*sum(self.algoNeuralNetwork.predict(self.validationData)==self.validationClases)/self.validationData.shape[1], 2), "%"
        confusionMatrix =  confusion_matrix(self.validationClases, self.algoNeuralNetwork.predict(self.validationData))
        utils.plot_confusion_matrix(confusionMatrix, classes=range(10), normalize=False, title='Neural Network Confusion matrix')
        plt.figure()    
        
        # load json and create model
        json_file = open('model_mnist.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.algoCNN = model_from_json(loaded_model_json)
        # load weights into new model
        self.algoCNN.load_weights("model_mnist.h5")
        self.algoCNN.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print "============ RESULTADOS CNN ============="
        y_predicted = np.argmax(self.algoCNN.predict(originalFullDataset[:, :validationSize].T.reshape(validationSize, 28, 28, 1)), axis=1)+1
        print "Validation: ", round(100*sum(y_predicted==self.validationClases)/self.validationData.shape[1], 2), "%"
        confusionMatrix =  confusion_matrix(self.validationClases, y_predicted)
        utils.plot_confusion_matrix(confusionMatrix, classes=range(10), normalize=False, title='Convolutional Neural Network Confusion matrix')
        plt.show()  
        
        
        self.setup()
        
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def done(self):
        self.c.postscript(file=self.FILE + ".eps",
                  colormode="gray")
        self.imageInMemory.save(self.FILE+".jpg")
        self.input = np.array(self.imageInMemory.resize((28,28), Image.ANTIALIAS)).flatten()
        #plt.imshow(self.input, cmap='gray')
        inputProcessed  = -self.input+255
        for preprocesador in self.preprocesor:
            inputProcessed = preprocesador.transform(inputProcessed)
        print "======== RESULTADOS ========="
        print "Perceptron         dice: ", self.algo.predict(inputProcessed) -1
        print "Minimos cuadrados  dice: ", self.algoLeastSquare.predict(inputProcessed) -1
        print "Fisher LDA         dice: ", self.algoLinearDiscriminantAnalysis.predict(inputProcessed) -1
        print "Neural Network     dice: ", self.algoNeuralNetwork.predict(inputProcessed) -1
        print "CNN                dice: ", np.argmax(self.algoCNN.predict( (self.input/255).reshape(1, 28, 28, 1)), axis=1)
        return 
        
    def use_eraser(self):
        self.imageInMemory = Image.new("L", (self.WIDTH, self.HEIGHT), 'white')
        self.drawInMemory = ImageDraw.Draw(self.imageInMemory)
        self.c.delete('all')

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = self.DEFAULT_COLOR
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.drawInMemory.line([self.old_x, self.old_y, event.x, event.y],
                               width=self.line_width)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

if __name__ == '__main__':
    start = Paint()