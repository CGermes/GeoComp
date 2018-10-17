# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:39:14 2018

@author: carlos
"""
from __future__ import division
import numpy as np
import time
from scipy import linalg

class Perceptron():
  def __init__(self, coef_aprendizaje=0.5, max_timeout=1):
    self.w = None
    self.MAX_SECONDS = max_timeout
    self.alpha = coef_aprendizaje
    
  def fit(self, X, t, random=False):
    K = max(t)   
    if random:
      w0 = np.random.uniform(size=(X.shape[0]+1,))  
    else:
      w0 = np.insert(np.zeros(X.shape[0]), 0, 1)
    X = np.insert(X, 0, 1, axis=0)
    self.T = self.arrayToMatrix(t)
    T = self.T
                
    wlist = []
    #Entrenamos los K perceptrones necesarios
    for k in range(0, K):
      w = np.copy(w0) 
      limit = time.time() + self.MAX_SECONDS  
      permutation = np.array(range(X.shape[1]))
      updated = True
      # Entrenamiento
      while updated:
        if random:        
          np.random.shuffle(permutation)
        updated = False
        # vuelta por todos los puntos
        for i in permutation:
          if w.dot(X[:, i])*T[k, i] < 0:
            w += self.alpha * X[:, i]*T[k, i]
            updated = True
        if time.time() > limit:
          updated = False
          print "No se termino con Clase: " +str(k)
          
      wlist.append(w)
    self.w = np.array(wlist)
 
  def predict(self, X): 
    if len(X.shape) == 1:
      X = X[:, np.newaxis]
    p = self.w.dot(np.insert(X, 0, 1, axis=0))
    return np.argmax(p, axis=0)+1
    
  def arrayToMatrix(self, vector):
    matrix = np.ones((max(vector), vector.size))*(-1)
    for i in range(0, vector.size):
      matrix[vector[i]-1, i] = 1
    return matrix
    
    
class Least_Squares():
  def __init__(self):
    self.w = None
    
  def fit(self, X, t):
    T = self.arrayToMatrix(t)
    X = np.insert(X, 0, 1, axis=0)
    self.w = np.linalg.inv(X.dot(X.T)).dot(X).dot(T.T)
  
  def predict(self, X):
    if len(X.shape) == 1:
      X = X[:, np.newaxis]
    p = self.w.T.dot(np.insert(X, 0, 1, axis=0))    
    return np.argmax(p, axis=0)+1
    
  def arrayToMatrix(self, vector):
    matrix = np.ones((max(vector), vector.size))*-1
    #matrix = np.zeros((max(vector), vector.size))
    for i in range(0, vector.size):
      matrix[vector[i]-1, i] = 1
    return matrix


class Linear_Discriminant_Fisher():
  def __init__(self):
    self.w = None
    self.c = None
    
  def fit(self, X, t):
    K = max(t)
    wlist = []
    clist = []
    #one vs all training
    for i in range(1, K+1):
      # Separamos los puntos en 2 conjuntos: de la clase y de fuera.
      X1 = X[:, np.where(t == i)[0]]
      X2 = X[:, np.where(t != i)[0]]
      # Calculamos medias de los conjuntos
      m1 = np.sum(X1, axis=1)/X1.shape[1]
      m2 = np.sum(X2, axis=1)/X2.shape[1]
      
      #Inicializamos la matriz de varianzas within classes
      Sw = np.zeros((X1.shape[0], X1.shape[0]))
      #Calculamos Sw
      aux = (X1-m1[:, np.newaxis])
      Sw = Sw + aux.dot(aux.T)
      aux = (X2-m2[:, np.newaxis])
      Sw = Sw + aux.dot(aux.T)
        
      # Matriz de la proyección
      w = np.linalg.solve(Sw, m2-m1)

      # medias y varianzas
      p1 = X1.shape[1]/X.shape[1]
      p2 = X2.shape[1]/X.shape[1]
      
      # Medias de los datos proyectados
      my1 = w.dot(m1)
      my2 = w.dot(m2)
      
      #Varianzas de los datos
      varianza1 = np.sum((w.dot(X1)-my1)**2) / X1.shape[1]
      varianza2 = np.sum((w.dot(X2)-my2)**2) / X2.shape[1]
      
      #Ecuacion para encontrar minimo (y maximo) en funcion del punto de corte c
      a0 = np.log(p1/np.sqrt(varianza1))-np.log(p2/np.sqrt(varianza2))
      a0 = a0 -(my1**2/(2*varianza1))+(my2**2/(2*varianza2))
      a1 = (my1/varianza1)-(my2/varianza2)
      a2 = (-1/(2*varianza1))+(1/(2*varianza2))
      #c = (-a1 + np.sqrt(a1**2 - 4*a2*a0)) / (2 * a2) #Maximo
      c = (-a1 - np.sqrt(a1**2 - 4*a2*a0)) / (2 * a2)
      
      wlist.append(w)
      clist.append(c)
    self.w = np.array(wlist)
    self.c = np.array(clist)
  
  def predict(self, X):  
    if len(X.shape) == 1:
      X= X[:, np.newaxis]
    p = self.w.dot(X) - self.c[:,np.newaxis]
    return np.argmin(p, axis=0)+1
    
    
class Multilayer_Perceptron():
    
  def __init__(self, n_inputs, list_hidden_layers, n_outputs, activation='sigmoid'):
        if not isinstance(list_hidden_layers, list):
          list_hidden_layers = [list_hidden_layers]
        
        self.layers_size = [n_inputs] + list_hidden_layers + [n_outputs]
        
        if activation == 'sigmoid':
          self.activation = self.sigmoid
          self.activationGradient = self.sigmoidGradient
        elif activation == 'tanh':
          self.activation = self.tanh
          self.activationGradient = self.tanhGradient
        elif activation == 'softmax':
          self.activation = self.softmax
          self.activationGradient = self.softmaxGradient
        
        self.layers=[]
        for i in range(1,len(self.layers_size)):        
          weight_matrix = np.random.uniform(size=(self.layers_size[i], self.layers_size[i-1]+1))   
          self.layers.append(weight_matrix) 
  
  def propagation(self, X):
    if len(X.shape) == 1:
      X = X[:, np.newaxis]
    self.z = [X]
    self.a = [X]
    
    # forward propagation (antes y despues de usar la funcion de activacion)
    for layer in self.layers:
      self.a.append( layer.dot( np.insert(self.z[-1], 0, 1, axis=0) ) )
      self.z.append( self.activation(self.a[-1]) )
    
    return self.z[-1]
    
  def back_propagation(self, X, y_expected):
    if len(X.shape) == 1:
      X = X[:, np.newaxis]
      y_expected  = y_expected[:, np.newaxis]
      
    N = float(X.shape[1])
    y_result = self.propagation(X)
    
    #Calculo y propagacion
    delta = [ y_result - y_expected ] 
    for i in range(1, len(self.layers)):
      a = np.insert(self.a[-1-i],0,1, axis=0)
      delta.append( np.multiply(self.activationGradient(np.insert(self.a[-1-i],0,1, axis=0)), self.layers[-i].T.dot(delta[-1]) )[1:, :] )
    delta.reverse()
    
    # Calculo los gradientes de cada capa
    gradients=[]
    for i in range(len(self.layers) ):
      gradients.append( (delta[i].dot(np.insert(self.z[i],0,1, axis=0).T))/N  )
    return gradients
  
  def fit(self, X, y_expected, learning_coef=0.1, iters=10000, batch_size=1):
    if len(X.shape) == 1:
      X = X[:, np.newaxis]
     
    y_expected = self.arrayToMatrix(y_expected)
    
    #Iter veces hacemos el descenson de gradiente
    N = X.shape[1]
    permutation = np.array(range(N)) 
    
    for j in range(iters):
      np.random.shuffle(permutation)
      batch_index = permutation[:batch_size]
      gradients=self.back_propagation(X[:, batch_index], y_expected[:,batch_index]) 
      for i in range(len(self.layers)):
        self.layers[i] = self.layers[i] - learning_coef*gradients[i]
        
  
  def predict(self, X):
    if self.layers_size[-1] == 1:
        return np.rint(self.propagation(X)[0])
    else :
        p=self.propagation(X)
        return np.argmax(p, axis=0)+1
    
    
  def sigmoid(self, value):
    return (1/ (1+np.exp(-value)) )
    
  def sigmoidGradient(self, value):
    evaluation = self.sigmoid(value)
    return np.multiply( evaluation, -evaluation +1 )
    
  def tanh(self, x):
      return np.tanh(x)
  
  def tanhGradient(self, x):
      return (1 - np.power(np.tanh(x), 2) )

  def softmax(self, x):
    maximum=np.max(x, axis=0)
    x_corrected= x-maximum
    exps = np.exp(x_corrected)
    return exps / np.sum(exps, axis=0)
  
  def softmaxGradient(self, x):
    evaluation = self.softmax(x)
    return np.multiply( evaluation, -evaluation +1 )

  def arrayToMatrix(self, vector):
    if self.layers_size[-1] == 1:
        return vector[np.newaxis, :]
    else :
        matrix = np.zeros((self.layers_size[-1], vector.size))
        for i in range(vector.size):
              matrix[vector[i]-1, i] = 1
        return matrix

if __name__ == '__main__':
  print "========================================"
  print "=    TESTING MULTILAYER PERCEPTRON     ="
  print "========================================"
  data=np.array([[0,0],[1,0],[0,1],[1,1]]).T
  clases = np.array([1, 2, 2, 1])
    
#  start=Multilayer_Perceptron(2, [4], 2)
#  start.fit(data, clases, 0.2, 10000)
  
#  start=Multilayer_Perceptron(2, [4], 1)
#  start.fit(data, clases-1, 0.5, 10000)    
    
#  start=Multilayer_Perceptron(2, [4, 2, 4], 2)
#  start.fit(data, clases, 0.3, 50000)

#  start=Multilayer_Perceptron(2, [4], 2, 'tanh')
#  start.fit(data, clases, 0.4, 10000)
   
  start=Multilayer_Perceptron(2, [4], 2, 'softmax')
  start.fit(data, clases, 0.4, 10000)
  
#  clases = np.array([1, 1, 2, 2])
#  start=Multilayer_Perceptron(2, [], 2, 'tanh')
#  start.fit(data, np.array([1, 1, 2, 2]), 1, 100)
  
  print start.propagation(data)
  resultado=  start.predict(data)
  print "Evlauacion de problema XOR (0,0), (1,0), (0,1), (1,1)"
  print "Reultado: ", resultado
  print "Expected: ", clases
  print "Coincidencia? ", np.array_equal(resultado, clases)

class Bayessian():        
    def fit(self, x, t):
        # Clasifica el punto x de R^D mediante el clasificador bayesiano.
        self.meanList = []
        self.sigmaList = []
        self.sigmaInvertList = []
        self.sizeList = []

        self.totalSize = x.shape[1]
        self.K = max(t)
        
        #estimación de la matriz de covarianza, sigmak    
        for i in range (1, self.K+1):
            # Puntos de la clase i
            Xi = x[:, np.where(t==i)[0]]
            # Puntos en la clase i
            ni=Xi.shape[1]
            # media de la clase i
            mi = np.sum(Xi, axis=1)/ni
            # puntos de i centrados en la media
            XiCentrada = Xi-mi[:, np.newaxis]
            # Varianza de la clase i            
            Sk = XiCentrada.dot(XiCentrada.T)/ni
            self.meanList.append(mi)
            self.sigmaList.append(np.linalg.det(Sk))
            self.sizeList.append(ni)
            self.sigmaInvertList.append(np.linalg.inv(Sk))
        return

        
    def predict(self, x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        
        k=[]
        #funcion que debemos minimizar
        for i in range(self.K):
            xCentered = x - self.meanList[i][:, np.newaxis]
            aux = np.diag(xCentered.T.dot(self.sigmaInvertList[i]).dot(xCentered))
            aux = aux + np.log(self.sigmaList[i])
            aux = aux -2*np.log(self.sizeList[i]/self.totalSize)
            k.append(aux)
        p = np.array(k)
        return np.argmin(p, axis=0)+1
        
if __name__ == '__main__':
  print "========================================"
  print "=    TESTING BAYESSIAN CLASSIFIER      ="
  print "========================================"
  data=np.array([[0],[1],[2],[3]]).T
  clases = np.array([1, 2, 2, 1])
   
  start=Bayessian()
  start.fit(data, clases)

  resultado=  start.predict(data)
  print "Evlauacion de problema XOR (0,0), (1,0), (0,1), (1,1)"
  print "Reultado: ", resultado
  print "Expected: ", clases
  print "Coincidencia? ", np.array_equal(resultado, clases)
  
  
  
################## PCA ##############################
class PCA():
  def __init__(self, epsilon=0.2):
    self.eps = epsilon

  def fit(self, X):
    #Calculamos la matriz de covarianza:
    self.xmedia = np.sum(X, axis=1)/X.shape[1]
    Xcentrada = X - self.xmedia[:, np.newaxis]
    S = Xcentrada.dot(Xcentrada.T)
    
    #Calculamos los autovalores y autovectores y los ordenamos
    autovalores, autovectores = np.linalg.eig(S)
    permutacion = np.argsort(autovalores)[::-1]
    autovectores = autovectores[:,permutacion]
    autovalores = autovalores[permutacion]
    
    #calculamos la dimension d idonea
    d = self.select_dimension(autovalores, self.eps)
    print "Dimension reducida: ", d
    print "Informacion descartada: ", sum(autovalores[d:])/sum(autovalores)
    
    #Recortamos la matriz de autovectores
    self.U = autovectores[:,:d]
    return
  
  def transform(self, X):
    if len(X.shape) == 1:
      X = X[:, np.newaxis]
    return self.U.T.dot(X-self.xmedia[:, np.newaxis])
    
  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)
    
  def select_dimension(self, autovalores, eps):
    limit = eps*sum(autovalores)
    s = sum(autovalores);
    d = 0;
    
    #incluimos eps dentreo de lo aceptable. esto es <=eps
    while (s >= limit):
        s = s-autovalores[d]
        d = d+1
        
    return d;
     
if __name__ == '__main__':
  
  print "========================================"
  print "=             TESTING PCA              ="
  print "========================================"
  from sklearn.datasets import fetch_mldata
  from sklearn.datasets import load_breast_cancer
  from sklearn.decomposition import PCA as PCA2
  bcwd = load_breast_cancer()
  mnist = fetch_mldata('MNIST original', data_home=".")
  
  print "==== cancer ====="
  trans = PCA(0.2)
  trans2 = PCA2(0.8)
  print "Dimesion original: ", bcwd.data.T[:, :].shape[0]
  cancer1 = trans.fit_transform(bcwd.data.T[:, :])
  cancer2 = trans2.fit_transform(bcwd.data[:, :]).T

  print "Dimension reducida nuestra: ", cancer1.shape[0]
  print "Dimension reducida sklearn: ", cancer2.shape[0]
  print "Dimensiones coinciden? ", cancer1.shape[0] == cancer2.shape[0]
  print "covarianza nuestras: ", np.cov(cancer1)
  print "covarianza sklearn : ", np.cov(cancer2)
  print "Covarianzas coinciden? ", np.all(np.round(np.cov(cancer1), 6) == np.round(np.cov(cancer2), 6))
  print "N se ha conservado? ", bcwd.data.T.shape[1] == cancer1.shape[1]
  
  
  print "==== digitos ====="
  print "Dimension original: ", mnist.data.T.shape[0]
  digitos1 = trans.fit_transform(mnist.data.T[:, :])
  digitos2 = trans2.fit_transform(mnist.data).T
  print "Dimension reducida nuestra: ", digitos1.shape[0]
  print "Dimension reducida sklearn: ", digitos2.shape[0]
  print "Dimensiones coinciden? ", digitos1.shape[0] == digitos2.shape[0]
  print "Covarianzas coinciden? ", np.all(np.round(np.cov(digitos1), 6) == np.round(np.cov(digitos2), 6))
  print "N se ha conservado? ", mnist.data.T.shape[1] == digitos1.shape[1]
  


################# LDA #########################
class LDA():
    def __init__(self, epsilon=0.2):
        self.eps = float(epsilon)
            
    def fit(self, X, T):
        self.K = np.max(T)
        D = X.shape[0]     
        self.N = X.shape[1]
        
        #Creamos una lista con los puntos separados por clases
        self.Xklist = []        
        for i in range (1, self.K+1):
            Xi = X[:, np.where(T==i)[0]]
            self.Xklist.append(Xi)
            
        #Creamos una lista con las medias de cada clase
        self.mklist = []
        for Xk in self.Xklist:
            mi = np.sum(Xk, axis=1)/Xk.shape[1]
            self.mklist.append(mi)
        
        #calculamos S_W como suma de los Sk
        Sw = np.zeros((D, D))
        for i in range (self.K):
            XkCentrada = self.Xklist[i]-self.mklist[i][:, np.newaxis]
            Sw = Sw + XkCentrada.dot(XkCentrada.T)
            
        #calculamos la media global
        m = np.sum(X, axis=1)/self.N        
        
        #Calculamos Sb
        Sb = np.zeros((D, D))
        for i in range (self.K):
            marray = self.mklist[i][:, np.newaxis] - m[:, np.newaxis]
            Sb = Sb + self.Xklist[i].shape[1] * marray.dot(marray.T)
            
        #esto es como hacer eigh(np.linalg.inv(Sw).dot(Sb))
        autovalores, autovectores = linalg.eigh(Sb, Sw)
        permutacion = np.argsort(autovalores)[::-1]
        autovectores = autovectores[:,permutacion]
        autovalores = autovalores[permutacion]
        autovectores = autovectores / np.linalg.norm(autovectores, axis=0)  
        
        #calculamos la dimension d idonea
        d = self.select_dimension(autovalores, self.eps)
        print "Dimension reducida LDA: ", d
        print "Informacion descartada LDA: ", sum(autovalores[d:])/sum(autovalores)
        
        #Recortamos la matriz de autovectores
        self.U = autovectores[:,:d]
        #print self.U.shape
        
        return
        
    def transform(self, X):
        if len(X.shape) == 1:
            X= X[:, np.newaxis]
        return self.U.T.dot(X)
     
    def fit_transform(self, X, T):
        self.fit(X, T)
        return self.transform(X)
    
#    def predict (self, x):
#        p = Bayessian(self.X, self.Xklist, self.mklist, self.K)
#        return p;
    
    def select_dimension(self, autovalores, eps):
        limit = eps*sum(autovalores)
        s = sum(autovalores);
        d = 0;
        #incluimos eps dentro de lo aceptable. esto es <=eps
        while (s >= limit):
            s = s-autovalores[d]
            d = d+1
        return d;
        
if __name__ == '__main__':
  print "========================================"
  print "=             TESTING LDA              ="
  print "========================================"
  from sklearn.datasets import fetch_mldata
  from sklearn.datasets import load_breast_cancer
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  bcwd = load_breast_cancer()
  mnist = fetch_mldata('MNIST original', data_home=".")
  
  print "==== Cancer ====="
  trans = LDA()
  trans2 = LinearDiscriminantAnalysis(solver='eigen', tol=0.2)
  
  print "Dimesion original: ", bcwd.data.T[:, :].shape[0]
  cancer1 = trans.fit_transform(bcwd.data.T[:, :], bcwd.target+1)
  cancer2 = trans2.fit_transform(bcwd.data[:, :], bcwd.target+1).T

  print "Dimension reducida nuestra: ", cancer1.shape[0]
  print "Dimension reducida sklearn: ", cancer2.shape[0]
  print "Dimensiones coinciden? ", cancer1.shape[0] == cancer2.shape[0]
  print "covarianza nuestras: ", np.cov(cancer1)
  print "covarianza sklearn : ", np.cov(cancer2)
  print "Covarianzas coinciden? ", np.round(np.cov(cancer1), 6) == np.round(np.cov(cancer2), 6)
  print "N se ha conservado? ", bcwd.data.T.shape[1] == cancer1.shape[1]