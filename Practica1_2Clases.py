from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import time

class Perceptron():
  MAX_SECONDS=10;
  def fit(self, X, t):    
    self.w = np.insert(np.zeros(X.shape[0]), 0, 1)
    X=np.insert(X,0,1, axis=0)
    limit = time.time() + self.MAX_SECONDS
    updated=True
    while updated :
      if time.time() > limit:
        break
      updated=False
      for i in range(0, X.shape[1]) :
        if self.w.dot(X[:,i])*t[i] < 0 :
          self.w += X[:,i]*t[i]
          updated = True

  def predict(self, x):  
    p = self.w.T.dot(np.insert(x,0,1, axis=0))
    return np.sign(p)

class Least_Squares():
  def fit(self, X, t):
    X=np.insert(X,0,1, axis=0)
    self.w= np.linalg.inv(X.dot(X.T)).dot(X).dot(t)
  
  def predict(self, x):    
    p = self.w.dot(np.insert(x,0,1, axis=0));
    p=np.sign(p)
    return p;


class Linear_Discriminant_Fisher():
  def fit(self, X, t):
    X1 = X[:,np.where(t==-1)[0]]
    X2 = X[:,np.where(t==1)[0]]
    m1 = np.sum(X1, axis=1)/X1.shape[1]
    m2 = np.sum(X2, axis=1)/X2.shape[1]  
    S = np.zeros((X1.shape[0],X1.shape[0]))
    for col in list((X1-m1[:,np.newaxis]).T):
      S += col[:, np.newaxis].dot(col[np.newaxis, :])
    for col in list((X2-m1[:,np.newaxis]).T):
      S += col[:, np.newaxis].dot(col[np.newaxis, :])
    self.w=np.linalg.inv(S).dot(m2-m1)
    p1=X1.shape[1]/X.shape[1]
    p2=X2.shape[1]/X.shape[1]
    sigma1sq=0
    sigma2sq=0    
    my1=self.w.dot(m1)
    my2=self.w.dot(m2)
    for dist in list((self.w.dot(X1)-my1)):
      sigma1sq += dist**2
    for dist in list((self.w.dot(X2)-my2)):
      sigma2sq += dist**2
    sigma1sq/=X1.shape[1]
    sigma2sq/=X2.shape[1]    
    a0=np.log(p1/np.sqrt(sigma1sq))-np.log(p2/np.sqrt(sigma2sq))-(my1**2/(2*sigma1sq))+(my2**2/(2*sigma2sq))
    a1=(my1/sigma1sq)-(my2/sigma2sq)
    a2=(-1/(2*sigma1sq))+(1/(2*sigma2sq))
    #self.c = (-a1 + np.sqrt(a1**2 - 4*a2*a0)) / (2 * a2)
    self.c = (-a1 - np.sqrt(a1**2 - 4*a2*a0)) / (2 * a2)
    
  
  def predict(self, x):    
    p = self.w.dot(x) - self.c;
    p=np.sign(p)
    return p;
    
    
    
class DataPoint(Circle):
  
  def __init__(self, xy, clase):
    self.clase = int(clase)   
    super(DataPoint, self).__init__(xy, 0.5, color=self.getColorClass())
  
  def getColorClass(self):
      if self.clase == -1:
        return 'red'
      else:
        return 'blue'
        
class CreatePoints():
    
    def __init__(self, fig, ax):
        self.algo=Least_Squares()
        self.border = None
        self.circle_list = []

        self.x0 = None
        self.y0 = None

        self.fig = fig
        self.ax = ax
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        
        self.cidpress = fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmove = fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

        self.press_event = None
        self.current_circle = None
        
        self.toolbar = QToolBar()
        self.fig.canvas.toolbar.parentWidget().addToolBarBreak()
        self.fig.canvas.toolbar.parentWidget().addToolBar(self.toolbar)      
        
        self.pbCalcular = QPushButton("Calcular")
        self.pbCalcular.clicked.connect(self.calcular)
        self.toolbar.addWidget(self.pbCalcular)
        
        self.checkAutoCalcular = QCheckBox("Auto")
        self.checkAutoCalcular.stateChanged.connect(self.calcularConfig)
        self.toolbar.addWidget(self.checkAutoCalcular)
        
        self.cbClase = QComboBox()
        self.cbClase.addItems(["Clase 1", "Clase 2"])
        self.toolbar.addWidget(self.cbClase)
        

        
        self.cbAlgoritmo = QComboBox()
        self.cbAlgoritmo.addItems(["Minimos Cuadrados", "LDA (Fisher)", "Perceptron"])
        self.cbAlgoritmo.currentIndexChanged.connect(self.algoSelection)
        self.toolbar.addWidget(self.cbAlgoritmo)
        
        self.checkMapValues = QCheckBox("Map")
        self.checkMapValues.stateChanged.connect(self.mapConfig)
        self.toolbar.addWidget(self.checkMapValues)
        
        self.mapPower = QSpinBox()
        self.mapPower.setValue(3)
        self.mapPower.setRange(2, 5)
        self.mapPower.setEnabled(False)
        self.toolbar.addWidget(self.mapPower)


    def on_press(self, event):
        if event.button == 3:
            self.fig.canvas.mpl_disconnect(self.cidpress)
            self.fig.canvas.mpl_disconnect(self.cidrelease)
            self.fig.canvas.mpl_disconnect(self.cidmove)
            points = [circle.center for circle in self.circle_list]
            clases = [circle.clase for circle in self.circle_list]
            print points
            print clases
            return points, clases

        x0, y0 = event.xdata, event.ydata
        if(x0==None or y0==None):
          return
        for circle in self.circle_list:
            contains, attr = circle.contains(event)
            if contains:
                self.press_event = event
                self.current_circle = circle
                self.x0, self.y0 = self.current_circle.center
                return
        c = DataPoint((x0, y0), self.getClass())#self.getClassColor())
        self.ax.add_patch(c)
        self.circle_list.append(c)
        self.current_circle = None
        if self.checkAutoCalcular.isChecked():     
          self.calcular()
        self.fig.canvas.draw()

    def on_release(self, event):
        self.press_event = None
        self.current_circle = None

    def on_move(self, event):
        if (self.press_event is None or
            event.inaxes != self.press_event.inaxes or
            self.current_circle == None):
            return
        
        dx = event.xdata - self.press_event.xdata
        dy = event.ydata - self.press_event.ydata
        self.current_circle.center = self.x0 + dx, self.y0 + dy
        if self.checkAutoCalcular.isChecked():     
          self.calcular()
        self.fig.canvas.draw()


    def getClass(self):
      #Two clases getClass
      if self.cbClase.currentIndex() == 0:
        return -1 
      else:
        return 1
    
    def calcular(self):
      cm=LinearSegmentedColormap.from_list("prueba",
                                           [[ 1.        ,  0.        ,  0.        ,  0.2        ],
                                            [ 0.        ,  0.        ,  1.        ,  0.2         ]]
                                            , 4)
      #borramos la frontera si habia una calculada
      if(self.border != None):
        for element in self.border.collections:
            element.remove()
      
      points = np.array([np.array(circle.center) for circle in self.circle_list]).T
      clases = np.array([circle.clase for circle in self.circle_list])    
      plot_x = np.linspace(*self.ax.get_xlim(), num=1000)
      X, Y = np.meshgrid(plot_x, plot_x)
      xygrid=np.c_[X.ravel(), Y.ravel()].T
      
      if self.checkMapValues.isChecked():
        points=self.mapToPoly(points, self.mapPower.value())
        xygrid=self.mapToPoly(xygrid, self.mapPower.value())
        
        
      self.algo.fit(points, clases)
        
      Z = self.algo.predict(xygrid).reshape(X.shape)
      self.border = ax.contourf(X, Y, Z, levels=[-1,0,1], cmap=cm)
      
    def calcularConfig(self):
      if self.checkAutoCalcular.isChecked():
        self.pbCalcular.setEnabled(False)
      else:
        self.pbCalcular.setEnabled(True)
        
    def mapConfig(self):
      if self.checkMapValues.isChecked():
        self.mapPower.setEnabled(True)
      else:
        self.mapPower.setEnabled(False)
    
    def algoSelection(self):
      self.algo = [Least_Squares(), Linear_Discriminant_Fisher(), Perceptron()][self.cbAlgoritmo.currentIndex()]
      
    def mapToPoly(self, X, p):
      mapedX = []
      for point in list(X.T):
	newVector = [];
	for i in range(1, p+1):
	    for j in range(0, i+1):
		newVector.append(point[0]**(i-j) * point[1]**j)
	mapedX.append(newVector)
      return np.array(mapedX).T
        

if __name__ == '__main__':

    fig = plt.figure()
    ax = plt.subplot(111)
    
    start = CreatePoints(fig, ax)
    plt.show()
