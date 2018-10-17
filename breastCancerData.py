# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:48:39 2018

@author: carlos
"""
from __future__ import division
from sklearn.datasets import load_breast_cancer
from algorithms import Perceptron, Least_Squares, Linear_Discriminant_Fisher, Multilayer_Perceptron, Bayessian
from algorithms import PCA, LDA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import utils

pcaTolerance=0.0    

print "==== LOADING CANCER ====="

bcwd = load_breast_cancer()
fullDataset = bcwd.data.T
fullDatasetClases =  bcwd.target+1

N = fullDataset.shape[1]
D = fullDataset.shape[0]
K = max(fullDatasetClases)

print "Dimension:  ", D
print "Datos:      ", N
print "Clases:     ", K

preprocesor = []
#preprocesor.append(PCA(0.01))
#fullDataset = preprocesor[0].fit_transform(fullDataset)

preprocesor.append(LDA(0.2))
fullDataset = preprocesor[-1].fit_transform(fullDataset, fullDatasetClases)

trans = preprocesor[-1].U.T[0]
trans = trans / sum(trans)
permutacion = np.argsort(np.abs(trans))[::-1]

labels = bcwd.feature_names[permutacion]
weights = trans[permutacion]

for i in range(30):
    print '{0:2d}   {1:+7f}    {2:4s}'.format(i, weights[i], labels[i])


D = fullDataset.shape[0]

# preparamos las particiones de validacion y entrenamiento haciendo una mezcla (los datos van ordenados por clases)
permutation = np.array(range(N))
np.random.shuffle(permutation)
validationSize=int(0.2*N)
fullDataset=fullDataset[:, permutation]
fullDatasetClases = fullDatasetClases[permutation]

# Cargamos las particiones haciendo el preprocesado
data= fullDataset[:,validationSize:]   
clases= fullDatasetClases[validationSize:]
validationData = fullDataset[:,:validationSize]
validationClases = fullDatasetClases[:validationSize]

#DEBUG Vemos que todas las clases estan representadas. Aunque se verá en la matriz de confusion
#        unique, counts = np.unique(validationClases, return_counts=True) 
#        print "Elementos en Validacion: ", dict(zip(unique, counts))

algoPerceptron=Perceptron()
print "Fitting with Perceptron"
algoPerceptron.fit(data, clases ) 
print "======= RESULTADOS PERCEPTRON ========"
print "Accuracy: ", round(100*sum(algoPerceptron.predict(data)==clases)/data.shape[1], 2), "%"
print "Validation: ", round(100*sum(algoPerceptron.predict(validationData)==validationClases)/validationData.shape[1], 2), "%"
confusionMatrix =  confusion_matrix(validationClases, algoPerceptron.predict(validationData))

utils.plot_confusion_matrix(confusionMatrix, classes=range(K), normalize=False, title='Perceptron Confusion matrix')
plt.figure()
plt.show()

algoLeastSquare=Least_Squares()
print "Fitting with Least Squares"
algoLeastSquare.fit(data, clases )
print "======= RESULTADOS LEAST SQUARES ========"
print "Accuracy: ", round(100*sum(algoLeastSquare.predict(data)==clases)/data.shape[1], 2), "%"
print "Validation: ", round(100*sum(algoLeastSquare.predict(validationData)==validationClases)/validationData.shape[1], 2), "%"
confusionMatrix =  confusion_matrix(validationClases, algoLeastSquare.predict(validationData))
utils.plot_confusion_matrix(confusionMatrix, classes=range(K), normalize=False, title='Minimos Cuadrados Confusion matrix')
plt.figure()      

algoLinearDiscriminantAnalysis=Linear_Discriminant_Fisher()
print "Fitting with Fisher LDA"
algoLinearDiscriminantAnalysis.fit(data, clases )
print "======= RESULTADOS FISHER LDA ========"
print "Accuracy:   ", round(100*sum(algoLinearDiscriminantAnalysis.predict(data)==clases)/data.shape[1], 2), "%"
print "Validation: ", round(100*sum(algoLinearDiscriminantAnalysis.predict(validationData)==validationClases)/validationData.shape[1], 2), "%"
confusionMatrix =  confusion_matrix(validationClases, algoLinearDiscriminantAnalysis.predict(validationData))
utils.plot_confusion_matrix(confusionMatrix, classes=range(K), normalize=False, title='LDA Fisher Confusion matrix')
plt.figure() 

algoNeuralNetwork=Multilayer_Perceptron(D, (D+K), K, 'softmax')
print "Fitting with Neural Network"
algoNeuralNetwork.fit(data, clases, 0.4, 100000, 1)
print "======= RESULTADOS NEURAL NETWORK ========"
print "Accuracy:   ", round(100*sum(algoNeuralNetwork.predict(data)==clases)/data.shape[1], 2), "%"
print "Validation: ", round(100*sum(algoNeuralNetwork.predict(validationData)==validationClases)/validationData.shape[1], 2), "%"
confusionMatrix =  confusion_matrix(validationClases, algoNeuralNetwork.predict(validationData))
utils.plot_confusion_matrix(confusionMatrix, classes=range(K), normalize=False, title='Neural Network Confusion matrix')
plt.figure()

algoBayessian=Bayessian()
print "Fitting with Fisher LDA"
algoBayessian.fit(data, clases )
print "======= RESULTADOS BAYESSIAN ========="
print "Accuracy:   ", round(100*sum(algoBayessian.predict(data)==clases)/data.shape[1], 2), "%"
print "Validation: ", round(100*sum(algoBayessian.predict(validationData)==validationClases)/validationData.shape[1], 2), "%"
confusionMatrix =  confusion_matrix(validationClases, algoBayessian.predict(validationData))
utils.plot_confusion_matrix(confusionMatrix, classes=range(K), normalize=False, title='Bayessian Confusion matrix')

plt.show()     

