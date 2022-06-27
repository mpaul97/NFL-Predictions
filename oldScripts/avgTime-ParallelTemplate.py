import pandas as pd
import numpy as np
import multiprocessing

pd.options.mode.chained_assignment = None

def q1(df):

	ps = pd.read_csv("volumePositionIDs_Info.csv")
	pl = pd.read_csv("info3QbData_02-21W16.csv")
	
	position = 'q'
	posCol = 'qb'

	depth = 1
	ii = depth - 1

	pl.drop(columns=['surface'], inplace=True)

	df_list = []

	for index, row in df.iterrows():
		abbr = row['name'].split(" - ")[0]
		wy = row['wy']
		pid = ps.loc[(ps['wy']==wy)&(ps['abbr']==abbr), posCol].values[0]
		ii = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr)].index.values[0]
		time = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr), 'time'].values[0]
		stats = pl.loc[(pl['p_id']==pid)&(pl['time']==time)&(pl.index<ii)]
		if depth == 1:
			for index1, row1 in stats.iterrows():
				starter = ps.loc[(ps['wy']==row1['wy'])&(ps['abbr']==row1['abbr']), posCol].values[0]
				if pid != starter:
					stats.drop(index1, axis=0, inplace=True)
		if stats.empty:
			stats = pl.loc[(pl['time']==time)&(pl.index<ii)].tail(20)
		stats = stats.tail(5)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/5)
		stats.insert(0, 'name', row['name'])
		stats.insert(1, 'p_id', pid)
		df_list.append(stats)

	new_df = pd.concat(df_list)
	new_df.drop(columns='time', inplace=True)
	new_df = new_df.fillna(new_df.mean())
	new_df = new_df.round(1)

	return new_df

def r1(df):

	ps = pd.read_csv("volumePositionIDs_Info.csv")
	pl = pd.read_csv("info3RbData_02-21W16.csv")
	
	position = 'r'
	posCol = 'rb1'

	depth = 1
	ii = depth - 1

	pl.drop(columns=['surface'], inplace=True)

	df_list = []

	for index, row in df.iterrows():
		abbr = row['name'].split(" - ")[0]
		wy = row['wy']
		pid = ps.loc[(ps['wy']==wy)&(ps['abbr']==abbr), posCol].values[0]
		ii = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr)].index.values[0]
		time = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr), 'time'].values[0]
		stats = pl.loc[(pl['p_id']==pid)&(pl['time']==time)&(pl.index<ii)]
		if depth == 1:
			for index1, row1 in stats.iterrows():
				starter = ps.loc[(ps['wy']==row1['wy'])&(ps['abbr']==row1['abbr']), posCol].values[0]
				if pid != starter:
					stats.drop(index1, axis=0, inplace=True)
		if stats.empty:
			stats = pl.loc[(pl['time']==time)&(pl.index<ii)].tail(20)
		stats = stats.tail(5)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/5)
		stats.insert(0, 'name', row['name'])
		stats.insert(1, 'p_id', pid)
		df_list.append(stats)

	new_df = pd.concat(df_list)
	new_df.drop(columns='time', inplace=True)
	new_df = new_df.fillna(new_df.mean())
	new_df = new_df.round(1)

	return new_df

def r2(df):

	ps = pd.read_csv("volumePositionIDs_Info.csv")
	pl = pd.read_csv("info3RbData_02-21W16.csv")
	
	position = 'r'
	posCol = 'rb2'

	depth = 2
	ii = depth - 1

	pl.drop(columns=['surface'], inplace=True)

	df_list = []

	for index, row in df.iterrows():
		abbr = row['name'].split(" - ")[0]
		wy = row['wy']
		pid = ps.loc[(ps['wy']==wy)&(ps['abbr']==abbr), posCol].values[0]
		ii = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr)].index.values[0]
		time = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr), 'time'].values[0]
		stats = pl.loc[(pl['p_id']==pid)&(pl['time']==time)&(pl.index<ii)]
		if depth == 1:
			for index1, row1 in stats.iterrows():
				starter = ps.loc[(ps['wy']==row1['wy'])&(ps['abbr']==row1['abbr']), posCol].values[0]
				if pid != starter:
					stats.drop(index1, axis=0, inplace=True)
		if stats.empty:
			stats = pl.loc[(pl['time']==time)&(pl.index<ii)].tail(20)
		stats = stats.tail(5)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/5)
		stats.insert(0, 'name', row['name'])
		stats.insert(1, 'p_id', pid)
		df_list.append(stats)

	new_df = pd.concat(df_list)
	new_df.drop(columns='time', inplace=True)
	new_df = new_df.fillna(new_df.mean())
	new_df = new_df.round(1)

	return new_df

def w1(df):

	ps = pd.read_csv("volumePositionIDs_Info.csv")
	pl = pd.read_csv("info3WrData_02-21W16.csv")
	
	position = 'w'
	posCol = 'wr1'

	depth = 1
	ii = depth - 1

	pl.drop(columns=['surface'], inplace=True)

	df_list = []

	for index, row in df.iterrows():
		abbr = row['name'].split(" - ")[0]
		wy = row['wy']
		pid = ps.loc[(ps['wy']==wy)&(ps['abbr']==abbr), posCol].values[0]
		ii = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr)].index.values[0]
		time = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr), 'time'].values[0]
		stats = pl.loc[(pl['p_id']==pid)&(pl['time']==time)&(pl.index<ii)]
		if depth == 1:
			for index1, row1 in stats.iterrows():
				starter = ps.loc[(ps['wy']==row1['wy'])&(ps['abbr']==row1['abbr']), posCol].values[0]
				if pid != starter:
					stats.drop(index1, axis=0, inplace=True)
		if stats.empty:
			stats = pl.loc[(pl['time']==time)&(pl.index<ii)].tail(20)
		stats = stats.tail(5)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/5)
		stats.insert(0, 'name', row['name'])
		stats.insert(1, 'p_id', pid)
		df_list.append(stats)

	new_df = pd.concat(df_list)
	new_df.drop(columns='time', inplace=True)
	new_df = new_df.fillna(new_df.mean())
	new_df = new_df.round(1)

	return new_df

def w2(df):

	ps = pd.read_csv("volumePositionIDs_Info.csv")
	pl = pd.read_csv("info3WrData_02-21W16.csv")
	
	position = 'w'
	posCol = 'wr2'

	depth = 2
	ii = depth - 1

	pl.drop(columns=['surface'], inplace=True)

	df_list = []

	for index, row in df.iterrows():
		abbr = row['name'].split(" - ")[0]
		wy = row['wy']
		pid = ps.loc[(ps['wy']==wy)&(ps['abbr']==abbr), posCol].values[0]
		ii = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr)].index.values[0]
		time = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr), 'time'].values[0]
		stats = pl.loc[(pl['p_id']==pid)&(pl['time']==time)&(pl.index<ii)]
		if depth == 1:
			for index1, row1 in stats.iterrows():
				starter = ps.loc[(ps['wy']==row1['wy'])&(ps['abbr']==row1['abbr']), posCol].values[0]
				if pid != starter:
					stats.drop(index1, axis=0, inplace=True)
		if stats.empty:
			stats = pl.loc[(pl['time']==time)&(pl.index<ii)].tail(20)
		stats = stats.tail(5)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/5)
		stats.insert(0, 'name', row['name'])
		stats.insert(1, 'p_id', pid)
		df_list.append(stats)

	new_df = pd.concat(df_list)
	new_df.drop(columns='time', inplace=True)
	new_df = new_df.fillna(new_df.mean())
	new_df = new_df.round(1)

	return new_df

def w3(df):

	ps = pd.read_csv("volumePositionIDs_Info.csv")
	pl = pd.read_csv("info3WrData_02-21W16.csv")
	
	position = 'w'
	posCol = 'wr3'

	depth = 3
	ii = depth - 1

	pl.drop(columns=['surface'], inplace=True)

	df_list = []

	for index, row in df.iterrows():
		abbr = row['name'].split(" - ")[0]
		wy = row['wy']
		pid = ps.loc[(ps['wy']==wy)&(ps['abbr']==abbr), posCol].values[0]
		ii = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr)].index.values[0]
		time = pl.loc[(pl['wy']==wy)&(pl['abbr']==abbr), 'time'].values[0]
		stats = pl.loc[(pl['p_id']==pid)&(pl['time']==time)&(pl.index<ii)]
		if depth == 1:
			for index1, row1 in stats.iterrows():
				starter = ps.loc[(ps['wy']==row1['wy'])&(ps['abbr']==row1['abbr']), posCol].values[0]
				if pid != starter:
					stats.drop(index1, axis=0, inplace=True)
		if stats.empty:
			stats = pl.loc[(pl['time']==time)&(pl.index<ii)].tail(20)
		stats = stats.tail(5)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/5)
		stats.insert(0, 'name', row['name'])
		stats.insert(1, 'p_id', pid)
		df_list.append(stats)

	new_df = pd.concat(df_list)
	new_df.drop(columns='time', inplace=True)
	new_df = new_df.fillna(new_df.mean())
	new_df = new_df.round(1)

	return new_df

def buildTime(position, depth):

	df = pd.read_csv("blue6CE.csv")

	#Parallel
	df_list1 = []

	num_cores = multiprocessing.cpu_count()-1
	num_partitions = num_cores
	df_split = np.array_split(df, num_partitions)

	if position == 'q':
		posCol = 'qb'
		tag = 'Qb' + str(depth)
		if __name__ == '__main__':
			pool = multiprocessing.Pool(num_cores)
			all_dfs = pd.concat(pool.map(q1, df_split))
			df_list1.append(all_dfs)
			pool.close()
			pool.join()
	elif position == 'r':
		posCol = 'rb' + str(depth)
		tag = 'Rb' + str(depth)
		if depth == 1:
			if __name__ == '__main__':
				pool = multiprocessing.Pool(num_cores)
				all_dfs = pd.concat(pool.map(r1, df_split))
				df_list1.append(all_dfs)
				pool.close()
				pool.join()
		else:
			if __name__ == '__main__':
				pool = multiprocessing.Pool(num_cores)
				all_dfs = pd.concat(pool.map(r2, df_split))
				df_list1.append(all_dfs)
				pool.close()
				pool.join()
	elif position == 'w':
		posCol = 'wr' + str(depth)
		tag = 'Wr' + str(depth)
		if depth == 1:
			if __name__ == '__main__':
				pool = multiprocessing.Pool(num_cores)
				all_dfs = pd.concat(pool.map(w1, df_split))
				df_list1.append(all_dfs)
				pool.close()
				pool.join()
		elif depth == 2:
			if __name__ == '__main__':
				pool = multiprocessing.Pool(num_cores)
				all_dfs = pd.concat(pool.map(w2, df_split))
				df_list1.append(all_dfs)
				pool.close()
				pool.join()
		else:
			if __name__ == '__main__':
				pool = multiprocessing.Pool(num_cores)
				all_dfs = pd.concat(pool.map(w3, df_split))
				df_list1.append(all_dfs)
				pool.close()
				pool.join()

	if df_list1:
		new_df = pd.concat(df_list1)
		new_df.to_csv("%s.csv" % ("timeAvg" + tag + "_05-21"), index=False)

############################################################
# CALLS

# buildTime('q', 1)

buildTime('r', 1)

buildTime('r', 2)

buildTime('w', 1)

buildTime('w', 2)

buildTime('w', 3)