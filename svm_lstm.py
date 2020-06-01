
import matplotlib.pyplot as plt
import seaborn as sns
from learn_module import LearnModule
import pickle
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.svm import SVC
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow import SparseTensor
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import docproc
import scipy as sci
import sparse
import numpy as np
import pandas as pd
import train_imdb
import random
import scipy.stats as sts

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

# Read database named as <twtf> with tweets 
def read_twt(twtf):

	f = open(twtf,"rb")
	twt = pickle.load(f)
	f.close()

	return twt

# Test svm produced in IMDB dataset on tweets
# FAILED
# (EXPERIMENT COUTING BY MUTUAL INFORMATION)
def test_svm():

	svm = train_imdb.load_svm_imdb()
	wvec = train_imdb.load_wvec_imdb()

	svmMod = LearnModule(svm,lambda mod,x:mod.predict(X))
	
	stonks = pd.read_csv("ABB-stonks.csv")
	twt = read_twt("ABB.twt")[:-1,1]
	twt = np.array(docproc.preproc(twt))
	
	vclosed = stonks.iloc[:,5].to_numpy()

	#Generate labels by the difference between
	#stonks value of t+1 and of t
	labs = vclosed[1:] - vclosed[:-1]
	idx_p = np.where(labs >= 0)[0]
	idx_n = np.where(labs < 0)[0]
	labs[idx_n] = 0
	labs[idx_p] = 1

	X = wvec.transform(twt)
	preds = svmMod.execute(X)
	
	return preds,labs

	
# Test svm produced in ABB.twt dataset on tweets
# FAILED
def test_svm_2(train_rate=0.7):

	svmMod = LearnModule(LinearSVC(penalty="l2",C=0.01),lambda mod,x:mod.predict(x))
	
	stonks = pd.read_csv("ABB-stonks.csv")
	twt = read_twt("ABB.twt")[:-1,1]

	twt = np.array(docproc.preproc(twt))
	
	vclosed = stonks.iloc[:,5].to_numpy()

	#Generate labels by the difference between
	#stonks value of t+1 and of t
	labs = vclosed[1:] - vclosed[:-1]
	idx_p = np.where(labs >= 0)[0]
	idx_n = np.where(labs < 0)[0]
	labs[idx_n] = 0
	labs[idx_p] = 1

	#Separating training and test
	idx_tr = np.random.choice(range(0,len(labs)),int(np.floor(train_rate*len(labs))))
	Xtr = twt[idx_tr]
	Ytr = labs[idx_tr]
	Xte = np.delete(twt,idx_tr)
	Yte = np.delete(labs,idx_tr)

	tfidf = TfidfVectorizer(ngram_range=(1,3))
	tfidf.fit(Xtr)
	feats = tfidf.transform(Xtr)	
	
	#TRAIN
	svmMod.train(feats,Ytr,lambda mod,x,y : mod.fit(x,y))
	preds = svmMod.execute(tfidf.transform(Xte))
	predstr = svmMod.execute(tfidf.transform(Xtr))
	
	return preds,Yte,predstr,Ytr
	
def embedd(x,d,t):

	xs = [x[ti*d-1:-(t-(ti-1))*d] for ti in range(1,t+1)]
	S = np.empty((len(xs[0]),t),dtype=float)

	i = 0
	for s in xs:
		S[:,i] = s
		i += 1		
		
	return S
	
def error_multipred_abs(real,pred,date_prev):

	err = []

	for i in range(0,len(pred)):
		err.append(np.absolute(np.sum((np.array(pred[i]) - np.array(real[i])))/date_prev))

	return np.sum(err)/len(pred)

def error_multipred_perprev_abs(real,pred):

	err = np.sum(np.absolute(np.array(pred) - np.array(real)),axis=0)/pred.shape[0]

	return err

def error_multipred_perday_abs(real,pred,date_prev):

	err = []

	for i in range(0,len(pred)):
		err.append(np.absolute(np.sum(np.array(pred[i]) - np.array(real[i]))/date_prev))

	return err

def error_multipred(real,pred,date_prev):

	err = []

	for i in range(0,len(pred)):
		err.append(np.sum((np.array(pred[i]) - np.array(real[i]))/date_prev))

	return np.sum(err)/len(pred)

def corr_multipred_perprev(real,pred):

	err = []

	for i in range(0,len(pred)):
		err.append(sts.spearmanr(np.array(pred[i]),np.array(real[i]))[0])

	return err

def error_multipred_perprev(real,pred):

	err = np.sum((np.array(pred) - np.array(real)),axis=0)/pred.shape[0]

	return err

def error_multipred_perday(real,pred,date_prev):

	err = []

	for i in range(0,len(pred)):
		err.append(np.sum(np.array(pred[i]) - np.array(real[i]))/date_prev)

	return err

def plot_multioutput_pred(real,pred):

	plt.plot(pred[:,pred.shape[1]-1],color="blue")
	plt.plot(real[:,real.shape[1]-1],color="red")	

	plt.show()
	plt.close() 

def plot_multioutput_imshow(real,pred):

	fig,ax = plt.subplots()

	ax.imshow(np.absolute(real-pred),aspect="auto")

	fig.show()
	fig.close() 
	
# Test svm produced in ABB stonks dataset considering time delay and embedding coordinates (see Taken's Theorem)
# @param	train_rate			Proportion in the entire set allocated for training
# @param	look_back			How many past days will be considered for training and prediction
# @param	delay				Delay coordinates considered
# @param	date_prev			How many days further the algorithm must predict
# @param	delay_prev			Delay coordinates considered for predictions
# @param	deg				Degree of polynomial SVR
# @return	Ytr,Yte,predstr,preds,lstmMod	Labels of training set," of test set, predicted labels of training set, " of test set, trained LearnModule
# preds = svm_lstm.test_svr_pred(train_rate=0.9,look_back=30,delay=1,deg=2,date_prev=4,delay_prev=7,C=0.001)
# preds = svm_lstm.test_svr_pred(train_rate=0.9,look_back=30,delay=1,deg=2,date_prev=7,delay_prev=1,C=0.001)
# preds = svm_lstm.test_svr_pred(train_rate=0.9,look_back=30,delay=1,deg=1,date_prev=7,delay_prev=1,C=0.001) 
def test_svr_pred(train_rate=0.7,look_back=7,delay=1,date_prev=7,delay_prev=1,deg=1,C=0.1):

	svmMod = LearnModule(MultiOutputRegressor(SVR(C=0.1,kernel="poly",degree=deg,verbose=True)),lambda mod,x:mod.predict(x))
	
	stonks = pd.read_csv("ABB-stonks.csv")
	vclosed = stonks.iloc[:,5].to_numpy()

	#Sliding window exponential average
	vclosed = pd.DataFrame.ewm(pd.DataFrame(vclosed),30).mean().to_numpy().reshape((1,len(vclosed))).tolist()[0]
	vclosed = list(map(lambda x:x,vclosed))
	#Embedding S = (S(0,t-t'd),S(d,t-(t'-1)d),...,S(t'd,t))
	emb = embedd(vclosed,delay,look_back)
	#Y = S[:,t']
	S = emb[:,:emb.shape[1]]
	labs = emb[:,emb.shape[1]-1]
	#Yt = (Y(t'd,t-t'd),Y(t'd,t-(t'-1)d),...,Y(t'd,t))
	labs = embedd(labs,delay_prev,date_prev+1)[:,1:]
	#Trim length of series in S onto length of series in Yt
	S = S[:labs.shape[0],:]
	Saux = S
	lseq = labs

	idx_tr = random.sample(range(labs.shape[0]),int(np.floor(train_rate*labs.shape[0])))
	idx_seq = range(int(np.floor(train_rate*labs.shape[0])),labs.shape[0])
	
	Ytr = labs[idx_tr,:]
	Str = S[idx_tr,:]
	#Test set
	Yte = np.delete(labs,idx_tr,axis=0)
	Ste = np.delete(S,idx_tr,axis=0)

	svmMod.train(Str,Ytr,lambda mod,x,y : mod.fit(x,y))
	#TESTS
	preds = svmMod.execute(Ste)

	predseq = []
	predseq = svmMod.execute(Saux[idx_seq,:])
		
	return Yte,preds,lseq[idx_seq,:],predseq,svmMod 
	
