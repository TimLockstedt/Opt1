from mat4py import loadmat
import numpy as np

# Kosten funktional
def J (x, dt, x_data):
    result = 0
    for i in range(len(x)):
        result += dt * np.abs(x[i] - x_data[i])**2
    return result

def f (d, v_max, model = "log"):
    if model == "log":
        return np.min([v_max * np.log(d), v_max], 1) # Limit v_max, wie im linearen modell
    else: # lin
        return v_max * (1 - 1/d)
	
def single_step(x, v_max, L, dt, model):
    N = len(x) # Number of initial values == number of vehicles in traffic
    dx = np.zeros(N) # momente initialisieren
    # Leader
    # dx[-1] = v_max
    dist = d(x)/L
    # for i in range(N):
    #         dx[i] = f(dist[i], v_max, model)
    dx = f(dist, v_max, model)
    return x + dt * dx, dx
    
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
	__xIt = np.ndarray
	__tIt = np.ndarray
	__ItError = np.ndarray
	__numx = np.ndarray	
	__J = np.ndarray
	__gesx = []
	__gesnumx = []
	__d = np.ndarray

	def get_d(self):
		return self.__d
	def get_numx(self):
		return self.__numx
	def get_J(self):
		return self.__J
	def get_gesx(self):
		return self.__gesx
	def get_gesnumx(self):
		return self.__gesnumx
	def get_len(self):
		return self.__len
	def get_x(self):
		return self.__x
	def get_t(self):
		return self.__t
	def get_tIt(self):
		return self.__tIt
	def get_xIt(self):
		return self.__xIt
	def get_ItError(self):
		return self.__ItError
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
	def next_dataset(self, dataset_num = None):
		if dataset_num == None:
			if self.__counter < self.__len - 1:
				self.__counter += 1
				self.setXYTFromData(self.__counter)
				return
			else:
				self.__counter += 1
				print(f"Error: Index {self.__counter} is out of range of the dataset with length {self.__len} !!!")
				return
		else:
			if dataset_num < self.__len - 1:
				self.__counter = dataset_num
				self.setXYTFromData(self.__counter)
				return
			else:
				self.__counter = dataset_num
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

	def J (self, x, dt, x_data):
		result = 0
		for i in range(len(x)):
			result += dt * np.abs(x[i] - x_data[i])**2
		return result
	
	def set_numx(self, dt, L, v_max, n_dataset = __counter, model = "log"):
		self.next_dataset(n_dataset) # auf den gewollten Datensatz wechseln
		self.LinInterplation(dt) # durchschnittlicher Fehler der Interpolation abfragen
		self.switch_iteration() # Die interpolierten werte als zu berücksichtigende Daten setzen
		self.__numx = np.zeros_like(self.get_x()) # Array definieren, in dem die numerisch bestimmten orte gespeichert werden
		self.__J = np.zeros(self.get_t().shape[0] - 1)
		for i in range(len(self.__J)+1):
			x_soll, t_soll = self.getCurrentCoords() # Aktuelle ort, zeit punkte der autos ausgeben
			if i != 0: # Falls wir schon einen Schritt gegangen sind
				self.__J[i-1] = self.J(x, dt, x_soll) # Loss berechnen
			else: # Falls der erste durchlauf
				x, t = x_soll, t_soll # Startpositionen initialisieren
			x, _ = self.single_step(x, v_max, L, dt, model) # x aktualisieren
			self.__numx[:,i] = x
			t += dt

		self.__gesx.append(self.__x)
		self.__gesnumx.append(self.__numx)
		return	self.__numx

	def f (self, d, v_max, model = "log"):
		if model == "log":
			return np.min([v_max * np.log(d), np.ones_like(d) * v_max], 0) # Limit v_max, wie im linearen modell
		else: # lin
			return v_max * (1 - 1/d)
	
	def single_step(self, x, v_max, L, dt, model):
		N = len(x) # Number of initial values == number of vehicles in traffic
		dx = np.zeros(N) # momente initialisieren
		# Leader
		# dx[-1] = v_max
		dist = self.d(x)/L
		# for i in range(N):
		#         dx[i] = f(dist[i], v_max, model)
		dx = self.f(dist, v_max, model)
		return x + dt * dx, dx


	def switch_iteration(self):
		copy_x = self.__x.copy()
		copy_t = self.__t.copy()
		self.__x = self.__xIt 
		self.__t = self.__tIt 
		self.__xIt = copy_x
		self.__tIt = copy_t

	def test_correl(self, limit = 0.03):	
		corell = np.zeros(self.__x.shape[0])
		for i, auto in enumerate(self.__x):
			corell[i] = 1 - np.corrcoef(self.__t, auto)[0, 1]
		if np.sum(corell >= limit) >= 1:
			print(f"ACHTUNG mindestenz einmal eine größere Abweichung als {limit}")
		self.__ItError = corell
		return corell


	def test_interpolation(self, dt):
		rel_fehler = np.zeros(self.__x.shape[0])
		for i, auto in enumerate(self.__x):
			for x, t in zip(auto, self.__t):
				index = np.where((self.__tIt >= (t - dt/2)) & (self.__tIt <= (t + dt/2)))[0]
				if len(index) != 0:
					abs_fehler = np.abs(x-self.__xIt[i, index])
					rel_fehler[i] = rel_fehler[i] + abs_fehler/np.abs(x)
		avg_fehler = rel_fehler/len(self.__t)
		if np.sum(avg_fehler >= 0.05) > 1:
			print("ACHTUNG mindestenz einmal eine größere Abweichung als 5%")
		self.__ItError = avg_fehler
		return avg_fehler

	def LinApprox(self, dt=0.001):
		if np.sum(self.test_correl() >= 0.03) >= 1:
			print(f"ACHTUNG mindestenz einmal eine größere Abweichung als {0.03}")
		
		ort = self.__x
		zeit = self.__t
		zeit_int = np.arange(zeit[0], zeit[-1], dt)
		ort_int = np.zeros((ort.shape[0], len(zeit_int)))
		for i, auto in enumerate(ort):
			steigung = (np.max(auto)-np.min(auto))/(np.max(zeit)-np.min(zeit))
			ort_int[i,:] = steigung * np.arange(0, np.max(zeit)-np.min(zeit), dt) + np.min(auto)

		self.__xIt = ort_int
		self.__tIt = zeit_int
		# self.test_interpolation(dt)
		return ort_int, zeit_int


	def LinInterplation(self, dt=0.001):
		if np.sum(self.test_correl() >= 0.03) >= 1:
			print(f"ACHTUNG mindestenz einmal eine größere Abweichung als {0.03}")
		
		ort = self.__x
		zeit = self.__t
		zeit_int = np.arange(zeit[0], zeit[-1], dt)
		ort_int = np.zeros((ort.shape[0],*zeit_int.shape))
		for i, auto in enumerate(ort):
			ort_int[i, :] = np.interp(zeit_int, zeit, auto)
		self.__xIt = ort_int
		self.__tIt = zeit_int
		# self.test_interpolation(dt)
		return ort_int, zeit_int
	
	def d(self, x:np.ndarray): # Abstandsfunktion
		d = np.ones_like(x) * np.inf # anfangs distanz auf sehr groß setzen
		order_mask = np.argsort(x) # nach i-ter achse sortieren, hier x achse -> 0
		x_ordered = x[order_mask]
		for i in range(len(x_ordered)-1):
			dist = np.linalg.norm(x_ordered[i] - x_ordered[i+1]) # Da die werte geordnet sind, muss nur die Distanz zum nächsten auto berücksichtigt werden
			d[i] = dist if d[i] > dist else d[i]
				
		reordered_d = d[np.argsort(order_mask)]
		return reordered_d

    
def d(x:np.ndarray): # Abstandsfunktion
	d = np.ones_like(x) * np.inf # anfangs distanz auf sehr groß setzen
	order_mask = np.argsort(x) # nach i-ter achse sortieren, hier x achse -> 0
	x_ordered = x[order_mask]
	for i in range(len(x_ordered)-1):
		dist = np.linalg.norm(x_ordered[i] - x_ordered[i+1]) # Da die werte geordnet sind, muss nur die Distanz zum nächsten auto berücksichtigt werden
		d[i] = dist if d[i] > dist else d[i]
			
	reordered_d = d[np.argsort(order_mask)]
	return reordered_d


def LinInterplation(data, dt=0.001):
	ort = data.get_x()
	zeit = data.get_t()
	zeit_int = np.arange(zeit[0], zeit[-1], dt)
	ort_int = np.zeros((ort.shape[0], len(zeit_int)))
	for i, auto in enumerate(ort):
		steigung = (np.max(auto)-np.min(auto))/(np.max(zeit)-np.min(zeit))
		ort_int[i,:] = steigung * np.arange(0, np.max(zeit)-np.min(zeit), dt) + np.min(auto)
	return ort_int, zeit_int

def test_interpolation(data, ort_int, zeit_int, dt):
	ort, zeit = data.get_x(), data.get_t()
	rel_fehler = np.zeros(ort.shape[0])
	for i, auto in enumerate(ort):
		for x, t in zip(auto, zeit):
			index = np.where((zeit_int >= (t - dt/2)) & (zeit_int <= (t + dt/2)))[0]
			if len(index) != 0:
				abs_fehler = np.abs(x-ort_int[i, index])
				rel_fehler[i] = rel_fehler[i] + abs_fehler/np.abs(x)
	avg_fehler = rel_fehler/len(zeit)
	if np.sum(avg_fehler >= 0.05) > 1:
		print("ACHTUNG mindestenz einmal eine größere Abweichung als 5%")
	return avg_fehler

def p1(dt, data:data_class, v_max, d):
	p1 = np.zeros_like(data.get_x()[0,:]) # anpassen
	t_arr = np.flip(data.get_t())
	for i, j in enumerate(t_arr):
		if i == 0:
			p = 0
		else:
			p = p + dt * 2*(data.get_numx()[0,-i]-data.get_x()[0,-i]) - dt * p * (v_max/d[-i])
		p1[-i] = p
	return p1

def p2_N_1(dt, data:data_class, pi_1, d, n, v_max): # n tes auto
	pi = np.zeros_like(data.get_x()[1:-1,:])[0] # anpassen
	t_arr = np.flip(data.get_t())
	for i, j in enumerate(t_arr):
		if i == 0:
			p = 0
		else:
			p = p + dt * 2*(data.get_numx()[n,-i]-data.get_x()[n,-i]) - dt * p * (v_max/d[-i]) + dt * pi_1[-i] * v_max/d[-i]
		pi[-i] = p
	return pi

def pN(dt, data:data_class, pi_1, d, v_max): # n tes auto
	pi = np.zeros_like(data.get_x()[1:-1,:])[0] # anpassen
	t_arr = np.flip(data.get_t())
	for i, j in enumerate(t_arr):
		if i == 0:
			p = 0
		else:
			p = p + dt * 2*(data.get_numx()[-1,-i]-data.get_x()[-1,-i]) + dt * pi_1[-i] * v_max/d[-i]
		pi[-i] = p
	return pi


def pGes(dt, data:data_class, v_max):
	darr = np.array([data.d(i) for i in data.get_x().T])
	pges = np.zeros_like(data.get_x())
	pges[0,:] = p1(dt, data, v_max, darr[:,0])
	for n in range(1, pges.shape[0]-1):
		pges[n,:] = p2_N_1(dt, data, pges[n-1,:], darr[:,n], n, v_max)
	pges[n+1,:] = pN(dt, data, pges[n,:], darr[:,-1], v_max)
	return pges

def integral(arg, dt):
    res = np.array([arg[i] * dt for i in range(arg.shape[0])])
    return np.sum(res)

def gradient(P_ges, d, v_max, L, dt):
    sk0 = integral(P_ges[-1] + np.sum(np.array([P_ges[i] * np.log(d[i]/L) for i in range(P_ges.shape[0]-1)]), axis=0), dt)
    sk1 = integral(np.sum(np.array([P_ges[i] * v_max/L for i in range(P_ges.shape[0]-1)]), axis=0), dt)
    return (sk0, sk1)