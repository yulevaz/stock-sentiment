
# Class which store instances and labels of the paper abstracts
#
class Dataset:

	def __init__(self,X,Y,date):
		self.X = X
		self.Y = Y
		self.date = date
		self.dict = None
		Sy = set(Y)
		dic = dict(zip(Sy,range(0,len(Sy))))
		self.numY = [dic[i] for i in Y]
		Sdt = set(date)
		dic = dict(zip(Sdt,range(0,len(Sdt))))
		self.numDate = [dic[i] for i in date]

	#GETS
	def getX(self):
		return self.X

	def getY(self):
		return self.Y
	
	def getNumY(self):
		return self.numY

	def getpX(self,i):
		return self.X[i,...]	

	def getpY(self,i):
		return self.Y[i]
	
	def getpNumY(self,i):
		return self.numY[i]

	def getDictionary(self):
		return self.dict
	
	def getDate(self):
		return self.date
	
	def getNumDate(self):
		return self.numDate

	def getpDate(self,i):
		return self.date[i]
	
	def getpNumDate(self,i):
		return self.numDate[i]


	#SETS
	def setX(self,X):
		self.X = X

	def setY(self,Y):
		self.Y = Y	

	def setNumY(self,Y):
		self.numY = Y	

	def setNumDate(self,date):
		self.numDate = date	

	def setpX(self,i,x):
		self.X[i,...] = x

	def setpY(self,i,y):
		self.Y[i] = y 
	
	def setpNumY(self,i,y):
		self.numY[i] = y
	
	def setpNumDate(self,i,date):
		self.numDate[i] = date

	def setDictionary(self,dictionary):
		self.dict = dictionary

	def setupDictionary(self):
		abst = self.X
		flatl = [item for sublist in abst for item in sublist]
		self.dict = set(flatl)
	
	def setDate(self,date):
		self.date = date

	def setpDate(self,i,date):
		self.date[i] = date

