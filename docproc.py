import dataset
import re
import pickle
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np

# Remove numbers and 1-2 letter "terms"
# @param 	abst	List of abstracts
# @return	list	List of abstracts preprocessed
#
def remove_num_nword(abst):
	
	filt = [re.sub(r'[^a-zA-Z]*',r'',i) for i in abst]	
	filt = [re.sub(r'^[a-zA-Z]{1,2}$',r'',a) for a in filt]
	filt = list(filter(None,filt))

	return filt	

# Load data from files
# @param	ftwt		Tweets file name
# @return	np.array	Dataset with preprocessed texts
#
def load_tweets(ftwt):

	f = open(ftwt,"rb")
	twts = pickle.load(f)
	f.close()
	stks = pd.read_csv(fstonks)
	twts = np.delete(twts,0,1)

	return twts
 
# Preprocess a list of abstracts (stop words removal and stemming)
# @param 	twts		List of tweets
# @return	dataset		Dataset of preprocessed abstracts
#
def preproc(twts):	

	stop_words = set(stopwords.words('english'))
	stab = []
	lstem = LancasterStemmer()
	REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
	REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
	REPLACE_REP = re.compile(r"(\w)\1{2,}")
	
	#abst = [i for sublist in dat for i in sublist]
	i = 0
	for tweets in twts:
		tweets = tweets.lower()
		tweets = re.sub(r'[^\w\s]','',tweets)
		tweets = REPLACE_NO_SPACE.sub("", tweets)
		tweets = REPLACE_WITH_SPACE.sub(" ", tweets)
		tweets = REPLACE_REP.sub("",tweets)
		word_tokens = word_tokenize(tweets)
		filtered_sentence = [lstem.stem(w) for w in word_tokens if not w in stop_words]
		stab.append(filtered_sentence)
		i += 1
		print(i/len(twts))
		

	stab = [remove_num_nword(i) for i in stab]	
	stab = list(map(lambda x : ' '.join(x),stab))
	return stab

# Write tfidf features from a list of documents
# @param	dsetf		Dataset object of preprocessed documents
# @param	featf		Files to save train features
# @param	ngram		ngram_range (combination) of a word
def save_train_features(dsetf,featf,ngram=(1,2)):

	# Reading dataset object
	f = open(dsetf,"rb")
	dset = pickle.load(f)
	f.close()

	X = list(map(lambda x : ' '.join(x),dset.getX()))
	tfidf = TfidfVectorizer(ngram_range=(1,2))
	tfidf.fit(X)
	feats = tfidf.transform(X)	
	
	# Writting tfidf vectorization object
	f = open(featf+".vec","wb")
	pickle.dump(tfidf,f)
	f.close()

	# Writting tfidf features object
	f = open(featf+".x","wb")
	pickle.dump(feats,f)
	f.close()

	#Writting labels for training
	f = open(featf+".y","wb")
	pickle.dump(dset.getNumY(),f)
	f.close()
	
