
from learn_module import LearnModule
import pickle
from sklearn.svm import LinearSVC
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow import SparseTensor
import scipy as sci
import numpy as np


# Transform a sparse numpy.array to SparseTensor from tensorflow,
# API does not accept a low level type I need to transform it 
# just because life is like that! gezzz
# @param	csr		numpy.array with sparse data instance		
# @return 	SparseTensor	A great well formed sparse tensor to fit all requirements for
#				tensorflow, that is it MA FRIEND! Bitchie function ever; it whisper bff in ma ears!	
#				REALLY! WHY I HAVE TO CAST THIS SHIT?! 
#				PATTERNIZE YOUR'R FUCKING BULLSHIT!
#				(sorry guys, I known the trade-off between pattern/efficiency)
#				BUT I WOULD HAVE HEARD THE FUCKING SAME SHIT!
#				(but yes, we are all polite!)
def to_sparse_tensor(csr):

	csr_data = sci.sparse.find(csr)
	ix = csr_data[0].reshape((csr_data[0].shape[0],1))
	iy = csr_data[1].reshape((csr_data[1].shape[0],1))
	csr_indices = np.hstack((ix,iy))
	csr_values = csr_data[2]
	csr_shape = [csr.shape[0],csr.shape[1]]
	sparse = SparseTensor(indices=csr_indices,values=csr_values,dense_shape=csr_shape)
	sparse = tf.sparse.reorder(sparse)

	return sparse

# This is that function that you have to create to read an machine learning architecture
# because some smart-ass (english language does not typically employ hyphen, but, that is quite a but, mind the impact when you adopt it 
# in "smart-ass", LEARN HOW TO HYPHENIZE YOUR WORDS DUMB-ASS!)
# Note that, still, I need two functions to read the architecture of my LEARNING MAšINA!!
# @param	fname			File name for sure, as always!
# @return 	tensorflow.keras model	Yeah, the best of the shitty ones! (It works really well, I'm a fan, but a full of bullsthit fan)
def load_mlp(fname):

	# load json and create model
	json_file = open(fname+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(fname+".h5")
	sgd = SGD(lr=0.1, momentum=0.9)
	loaded_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return loaded_model

# Load fucking model!
# @param	fname			String with the name of the fucking model!
# @return	pandas probably		PANDAS DATASET MUNFUKER! DRUNKEDD! 
def load_svm(fname):
	
	f = open(fname,"rb")
	mod = pickle.load(f)
	f.close()
	return mod
i
# Neither do I known what I really did in this function!
# Kidding, I know, IT'S A FUCKIN ENSEMBLE CATEGORY! (This is mathematically wrong, exists a learning categories
# which comprises "singletons" algorithms such as a simple SVM, MLP train-test-validation scheme whose procedures
# will be always based on > consume data > adapt somehow > respond measure on data as user defines it). ]
# I WILL EXPLAIN THIS IN THE FOLLOWING CODE!!!!!
def get_ensemble():

	svm_mname = "svm_model.skl"
	nn_mname = "nn_model"

	#SVM MODEL
	svm = LearnModule()
	#It is out of order, sorry
	#Prediction, measures, they are up to the user, choose as a lambda function
	#Creating a model from a file, given all variations among APIs, they are up to the user
	#I AM A USER I WANT FIRST LAMBDA AS A MACHINE LERNING EVALUATION MEASURE!!!!
	#the other parameter read MA MODELAAAAAA!!!!!
	svm.buildFromFile(lambda mod,x : mod.predict(x),lambda : load_svm(svm_mname))

	#MLP model
	#Same shit
	mlp = LearnModule()
	mlp.buildFromFile(lambda mod,x : mod.predict(to_sparse_tensor(x)),lambda : load_mlp(nn_mname))

	W = [2,1]
	#Ensemble model
	#Now things get real dirty, have you ever though ensemble learning machines as a single learning machine?
	#Why took a weighted averge between learning machine results can't be defined as a learning model?
	#IT FUCKING CAN BE DEFINED AS A LEARNING MODEL!!!!!!
	ens = LearnModule([svm,mlp],lambda mod,x : list(map(lambda mod,y=x : mod.execute(y),mod))/3)

	return ens
	
