from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from algorithms import Perceptron, Least_Squares, Linear_Discriminant_Fisher, Bayessian, Multilayer_Perceptron

GRID_DETAIL=100

class DataPoint(Circle):
  
  def __init__(self, xy, clase):
    self.clase = int(clase)   
    super(DataPoint, self).__init__(xy, 0.5, color=self.getColorClass())
  
  def getColorClass(self):
      return ['red', 'blue', 'green', 'black', 'darkorange'][self.clase-1]
        
class CreatePoints():
    
    def __init__(self, fig, ax):
        self.algo = Least_Squares()
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
        self.cbClase.addItems(["Clase 1", "Clase 2", "Clase 3", "Clase 4", "Clase 5"])
        self.toolbar.addWidget(self.cbClase)
        

        
        self.cbAlgoritmo = QComboBox()
        self.cbAlgoritmo.addItems(["Least Square", "LDA Fisher", "Perceptron", "Bayessian", "Neural Net"])
        self.cbAlgoritmo.currentIndexChanged.connect(self.algoSelection)
        self.toolbar.addWidget(self.cbAlgoritmo)
        
        self.checkXorSpecial = QCheckBox("XOR")
        self.toolbar.addWidget(self.checkXorSpecial)
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
        if x0 == None or y0 == None:
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
      return self.cbClase.currentIndex()+1

    
    def calcular(self):
      cm = LinearSegmentedColormap.from_list("prueba",
                                           [[ 1.        ,  0.        ,  0.        ,  0.2        ],
                                            [ 0.        ,  0.        ,  1.        ,  0.2        ],
                                            [ 0.        ,  0.50196078,  0.        ,  0.2        ],
                                            [ 0.        ,  0.        ,  0.        ,  0.2        ],
                                            [ 1.        ,  0.54901961,  0.        ,  0.2        ]]
                                            , 5)
      #borramos la frontera si habia una calculada
      if self.border != None:
        for element in self.border.collections:
            element.remove()
      
      points = np.array([np.array(circle.center) for circle in self.circle_list]).T
      clases = np.array([circle.clase for circle in self.circle_list])    
      plot_x = np.linspace(*self.ax.get_xlim(), num=GRID_DETAIL)
      X, Y = np.meshgrid(plot_x, plot_x)
      xygrid = np.c_[X.ravel(), Y.ravel()].T
      
      if self.checkMapValues.isChecked():
        points = self.mapToPoly(points, self.mapPower.value())
        xygrid = self.mapToPoly(xygrid, self.mapPower.value())
      elif self.checkXorSpecial.isChecked():
        points = self.mapXorSpecial(points)
        xygrid = self.mapXorSpecial(xygrid)
        
      self.algo.fit(points, clases)
        
      Z = self.algo.predict(xygrid).reshape(X.shape)
      self.border = self.ax.contourf(X, Y, Z, levels=[0, 1, 2, 3, 4, 5], 
                                     cmap=cm, antialiased=False)
      
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
      self.algo = [Least_Squares(), Linear_Discriminant_Fisher(), Perceptron(),\
      Bayessian(), Multilayer_Perceptron(2, 10, 5, 'softmax')]\
      [self.cbAlgoritmo.currentIndex()]
      if self.cbAlgoritmo.currentIndex() == 4:
          self.checkXorSpecial.setEnabled(False)
          self.checkMapValues.setEnabled(False)
          self.checkXorSpecial.setChecked(False)
          self.checkMapValues.setChecked(False)
      else:
          self.checkXorSpecial.setEnabled(True)
          self.checkMapValues.setEnabled(True)
          
      
    def mapToPoly(self, X, p):
      mapedX = []
      for point in list(X.T):
        newVector = [];
        for i in range(1, p+1):
          for j in range(0, i+1):
            newVector.append(point[0]**(i-j) * point[1]**j)
        mapedX.append(newVector)
      return np.array(mapedX).T
      
    def mapXorSpecial(self, X):
      mapedX = []
      for point in list(X.T):
        mapedX.append([point[0]**2, 2*point[0]*point[1], point[1]**2])
      return np.array(mapedX).T
        

if __name__ == '__main__':

    fig = plt.figure()
    ax = plt.subplot(111)
    
    start = CreatePoints(fig, ax)
    plt.show()
