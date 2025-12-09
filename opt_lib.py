from mat4py import loadmat
import numpy as np

# Kosten funktional
def J (x, dt, x_data):
    result = 0
    for i in range(len(x)):
        dt * np.abs(x[i] - x_data[i])**2
    return result

def f (d, v_max, model = "log"):
    if model == "log":
        return np.min([v_max * np.log(d), v_max]) # Limit v_max, wie im linearen modell
    else: # lin
        return v_max * (1 - 1/d)
    
# Den mittlwert von J jederzeit ausgeben können:
class J_mittel():
    __val = 0
    __n = 0
    def set_val(self, x):
        self.__val = x
        return
    def set_n(self, x):
        self.__n = x
        return
    
    def get_val(self):
        return self.__val
    def get_n(self):
        return self.__n
    
    def get_res(self):
        return self.__val/self.__n

    def __init__(self, x):
        self.__val = x
        self.__n = 1
        return
    
    def add(self, x):
        self.__val += x
        self.__n += 1
        return
    

class data_class():
	__data = []
	__len = 0
	__x = np.ndarray
	__t = np.ndarray
	__counter = 0
	__unterCounter = 0

	def get_x(self):
		return self.__x
	def get_t(self):
		return self.__t
	def set_x(self, x):
		self.__x = x
		return 
	def set_t(self, x):
		self.__t = x
		return
	def set_counter(self, x):
		self.__counter = x
		return
	def __init__(self, path):
		self.__data = loadmat(path)["sequences"][:]
		self.__len = len(self.__data)
		self.__counter = 0
		self.setXYTFromData(self.__counter)
		return
	def setXYTFromData(self, x):
		self.__x = np.array(self.__data[x][0].get("Xarr"))
		self.__t = np.array(self.__data[x][0].get("Tarr"))
		return 
	def next_dataset(self):
		if self.__counter < self.__len - 1:
			self.__counter += 1
			self.setXYTFromData(self.__counter)
			return
		else:
			self.__counter += 1
			print(f"Error: Index {self.__counter} is out of range of the dataset with length {self.__len} !!!")
			return
	def getCurrentCoords(self, i=None):
		if i == None:
			i = self.__unterCounter
		maxZeilen = self.__x.shape[0]
		maxSpalten = self.__x.shape[1]
		xVals = np.array([self.__x[j, i] for j in range(maxZeilen) if i < maxSpalten])
		tVal = self.__t[i] if i < maxSpalten else None

		if i == self.__unterCounter:
			self.__unterCounter += 1
		return xVals, tVal

    
def d(x:np.ndarray): # Abstandsfunktion
	d = np.ones_like(x) * np.inf # anfangs distanz auf sehr groß setzen
	order_mask = np.argsort(x) # nach i-ter achse sortieren, hier x achse -> 0
	x_ordered = x[order_mask]
	for i, x_i in enumerate(x_ordered):
		for j, x_j in enumerate(x_ordered[i+1:]): # nur Werte mit größerem Index (i+1) sind für den vergleich relevant, da nur diese vor dem Auto sind.
			dist = np.linalg.norm(x_j - x_i)
			d[i] = dist if d[i] > dist else d[i]
	return d