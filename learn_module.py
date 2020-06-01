
# Class endowing Machine Learning modules for arbitrary tasks

class LearnModule:

	# Constructor
	# @param	model		The trained model of a ML technique
	# @param 	predict		The prediction function, given  
	#				in the form lambda model,x : (model,x)	
	def __init__(self,model=None,predict=None):

		self._model = model
		self._predict = predict 

	# Construct object from a file containing the trained model
	# @param	reader		A lambda function with a method for reading  
	# @param	predict		The evaluation function, given by a function 
	#				in the form lambda model,x : (model,x)	
	# @param	mname		Module name
	def buildFromFile(self,predict,reader):

		self._model = reader()
		self._predict = predict
		
	# Write model
	# @param	writer		A lambda function which writes, in the form				
	#				the model produced by the function self.train()
	def buildToFile(self,writer):
		
		writer(self._model)	

	#ENGINE

	# Execute predictions
	# @param	X		New instances
	# @return	numpy.array	Predictions
	def execute(self,X):

		val = self._predict(self._model,X)
		return val		

	# Train model
	# @param	X		Instances
	# @param	Y		Labels
	# @param	fitm		The training function in the
	#				form lambda model,x,y : model.train(x,y)	
	def train(self,X,Y,fitm):

		fitm(self._model,X,Y)						

	#GETS
	def getModel(self):
		return self._model

	def getPredictFun(self):
		return self._predict

	#SETS
	def setModel(self,model):
		self._model = model

	def setPredictFun(self,predict):
		self._predict = predict
