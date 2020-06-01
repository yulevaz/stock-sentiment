
import quandl
import datetime as dt
import numpy as np
import pandas as pd

quandl.ApiConfig.api_key="85TpzkrHR2mz5peps_mj"

# Get stonks data
# @param	qcode		Quandl code for company stonks
# @param	init_d		Initial date to be considered
# @param	last_d		Last date
# @return	numpy.array	Array with stonks retrieved
def get_stonks(qcode,init_d,last_d):

	data = quandl.get(qcode, start_date = init_d.strftime("%Y-%m-%d"), end_date = last_d.strftime("%Y-%m-%d"), returns="numpy", paginate=True)

	f = lambda x : list(x)
	data = np.array(list(map(f,data)))

	f = lambda x : pd.to_datetime(x[0])
	data[:,0] = list(map(f,data))

	return data

# Adjust table to consider weekends with constant stonks values
# @params	stonks_dt	Stonks data table
# @return	numpy.array	Table with the missing dates with constant stonks values
def adjust_table(stonks_dt):

	one_d = dt.timedelta(days=1)
	
	first_inst = stonks_dt[0,:]
	last_inst = stonks_dt[stonks_dt.shape[0]-1,:]
	first_d = first_inst[0]
	last_d = last_inst[0]

	inst_d = first_d

	ret_stonks = np.empty((stonks_dt.shape[0],stonks_dt.shape[1]-1))
	ret_date = np.empty((stonks_dt.shape[0],),dtype=dt.datetime)
	
	j = 0

	for i in range(1,stonks_dt.shape[0]):

		curr_inst = stonks_dt[i-1,:]
		next_inst = stonks_dt[i,:]
		next_d = next_inst[0] 
		aux_d = inst_d+one_d

		#add new dates with CONSTANT STONKS! (BECAUSE STONKSTOWN IS CLOSED MA FRIEND)
		while aux_d < next_d and j < ret_stonks.shape[0]:

			str_dt = aux_d
			ret_date[j] = str_dt
			ret_stonks[j,:] = curr_inst[1:]
			aux_d += one_d
			j += 1

		inst_d = next_d

	ret_stonks = np.hstack((ret_date.reshape(ret_date.shape[0],1),ret_stonks))
	ret_stonks = ret_stonks[:j,:]
	ret_stonks = np.vstack((stonks_dt,ret_stonks))
	ret_stonks = np.array(sorted(ret_stonks, key= lambda x: x[0]))

	return ret_stonks	
