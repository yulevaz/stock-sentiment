import pickle
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import scipy as sci
import docproc

# Read IMDB dataset, train a SVM and return it
# @return	sklearn.LinearSVC	svm model
def train_svm_imdb():

	f = open("imdb.x","rb")
	X = pickle.load(f)
	f.close()

	f = open("imdb.y","rb")
	Y = pickle.load(f)
	f.close()

	svm = LinearSVC()
	svm.fit(X,Y)
	
	return svm

# Save a SVM(IMDB) model to a file
# @param	svm	SVM model
def save_svm_imdb(svm):
	
	f = open("imdb_trained.svm","wb")
	pickle.dump(svm,f)
	f.close()

# Load the SVM(IMDB) model from file
# @return	sklearn.LinearSVC	svm model
def load_svm_imdb():

	f = open("imdb_trained.svm","rb")
	svm = pickle.load(f)
	return svm

# Load tfdif word vectorizer adopted in SVM training
# @return 	TfidfVectorizer		Tfidf Vectorizer
def load_wvec_imdb():

	f = open("imdb.vec","rb")
	vec = pickle.load(f)
	f.close()	
	
	return vec
