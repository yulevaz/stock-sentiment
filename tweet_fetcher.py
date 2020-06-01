
import GetOldTweets3 as got
import datetime as dt
import pandas as pd
import numpy as np
import time


# Fetching tweets
# @param	query		Search query
# @param	init_d		Initial date adopted to fetch the tweets
# @param	last_d		Last date
# @param	num_t		Number of tweets	
# @return	tweet		List of tweet objects
def get_tweets(query,init_d,last_d,lang,num_t):

	t0 = init_d.strftime("%Y-%m-%d")
	tf = last_d.strftime("%Y-%m-%d")
	print(t0)
	print(tf)
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(t0).setUntil(tf).setTopTweets(True).setMaxTweets(num_t)
	tweet = got.manager.TweetManager.getTweets(tweetCriteria)
	print(len(tweet))
	f = lambda x : x.text
	X = list(map(f,tweet))
	X = ' '.join(X)
	return X

# Fetching tweets by day
# @param	query		Search query
# @param	t0		Initial date adopted to fetch the tweets
# @param	tf		Last date
# @param	lang		Language of tweets
# @param	num_t		Number of tweets
# @return	list		A list with all tweets by day	
def tweets_by_day(query,t0,tf,lang,num_t):

	delta_t = dt.timedelta(days=1)

	twts = []

	while t0 < tf:
		try:
			tw = get_tweets(query,t0,t0+delta_t,lang,num_t)
			twts.append([t0,tw])
			print(twts)
			t0 += delta_t
		except:
			print("Oops!", sys.exc_info()[0], "occured.")
		finally:
			time.sleep(10)		
			tw = get_tweets(query,t0,t0+delta_t,lang,num_t)
			twts.append([t0,tw])
			print(twts)
			t0 += delta_t
			

	return np.array(twts)
