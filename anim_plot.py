import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd
import pickle

def set_colors(labs):
	
	c1 = 'r'#np.array([0,0,1])
	c2 = 'b'#np.array([1,0,0])

	#arr = np.array([],dtype=np.int64).reshape(0,3)
	arr = []
	
	for lb in labs:
		if (lb == 0):
			#arr = np.vstack([arr,c1])
			arr += c1
		else:
			#arr = np.vstack([arr,c2])
			arr += c2

	return arr	

def plot():
	yrs = ["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]
	nfr = len(yrs) # Number of frames
	fps = 1 # Frame per sec


	xs = []
	ys = []
	zs = []
	ls = []

	for i in yrs: 
		fi = open(i+".proj","rb")
		projs = pickle.load(fi)
		fi.close()

		xs.append(projs[:,0])
		ys.append(projs[:,1])
		zs.append(projs[:,2])

		fi = open(i+".labs","rb")
		labs = pickle.load(fi)
		fi.close()
		ls.append(set_colors(labs))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	sct, = ax.plot([], [], [], "o", markersize=1)
	def update(ifrm, xa, ya, za, la, yr):
		#[sct.set_color(la[ifrm][i]) for i in range(0,len(la[ifrm]))]
		#sct.set_color(la[ifrm])
		if (yr[ifrm] == '2000'):
			ax.clear()
			ax.set_xlim(-2,2)
			ax.set_ylim(-2,2)
			ax.set_zlim(-2,2)
		ax.set_title(yr[ifrm])
		#sct.set_data(xa[ifrm], ya[ifrm])
		#sct.set_3d_properties(za[ifrm])
		ax.scatter(xa[ifrm],ya[ifrm],za[ifrm],c=la[ifrm],s=1)

	ani = animation.FuncAnimation(fig, update, nfr, fargs=(xs,ys,zs,ls,yrs), interval=1000/fps)

	plt.show()
