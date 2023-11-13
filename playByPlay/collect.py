import pandas as pd
import numpy as np
import urllib.request
import os
import time
import regex as re

from ordered_set import OrderedSet

class Player:
	def __init__(self, name, position):
		self.name = name
		self.position = position

def find_all(mystr, sub):
    start = 0
    while True:
        start = mystr.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def writeTable(url, key):

	try:

		fp = urllib.request.urlopen(url)
		mybytes = fp.read()

		mystr = mybytes.decode("utf8", errors='ignore')
		fp.close()

		start0 = mystr.index('id="div_pbp')

		mystr = mystr[start0:]

		# f = open("temp.txt", "w")
		# f.write(mystr)
		# f.close()

		# filepath = "temp.txt"
		# with open(filepath, 'r') as f:
		# 	dfs = pd.read_html(f.read(), match='Full Play-By-Play Table')

		dfs = pd.read_html(mystr, match='Full Play-By-Play Table')

		df = dfs[0]

		# convert/compress table
		for index, row in df.iterrows():
			qr = row['Quarter']
			if type(qr) == str:
				if 'Quarter' in qr or 'Regulation' in qr:
					df.drop(index, inplace=True)

		# df.to_csv("testPbpTable.csv", index=False)
		df.to_csv("data/all/" + key + ".csv.gz", index=False, compression="gzip")

		# if os.path.exists("temp.txt"):
		# 	os.remove("temp.txt")
		# else:
		# 	print("temp.txt -> Not Found")

	except ValueError:
		print("PBP not found at:", url)

	return

def decompress(key):

    df = pd.read_csv("pbpTables/" + key + ".csv.gz", compression="gzip")

    print(df.head(5)['Detail'])
 
    return

def scrape():

    df = pd.read_csv("%s.csv" % ("../data/gameData"))

    # pbp tracked from 1994 and onwards

    # start = df.loc[df['wy'].str.contains('1994')].index[0]

    # df = df.loc[df.index>=start]

    _dir = "data/all/"
    keys = [fn.replace('.csv.gz', '') for fn in os.listdir(_dir)]
    
    df = df.loc[~df['key'].isin(keys)]

    for index, row in df.iterrows():
        key = row['key']
        print(row['wy'])
        url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
        writeTable(url, key)
        time.sleep(2)
        
    return

def merge():
    
    df = pd.read_csv("%s.csv.gz" % "data/all/199409040buf", compression='gzip')
    
    cols = [col.lower() for col in df.columns]
    
    cols[-4:-2] = ['away_points', 'home_points']
    
    df_list = []
    
    fns = os.listdir('data/all/')
    
    for index, fn in enumerate(fns):
        print(index, len(fns))
        temp_df = pd.read_csv(('data/all/' + fn), compression='gzip')
        temp_df.columns = cols
        temp_df.insert(0, 'key', fn.replace('.csv.gz',''))
        temp_df.insert(1, 'num', [i for i in range(len(temp_df.index))])
        df_list.append(temp_df)
        
    new_df = pd.concat(df_list)
    
    new_df.to_csv("%s.csv" % "data/rawTrain", index=False)
    
    return

def createInfo():
    
    # 202201150buf (bad) 202201090tam (good)
    
    # good = pd.read_csv("%s.csv.gz" % "data/all/202201150buf", compression='gzip')
    # bad = pd.read_csv("%s.csv.gz" % "data/all/202201090tam", compression='gzip')
    
    # print(good.columns, bad.columns)
    # print(good.columns[-3], bad.columns[-3])
    # # good = -3 is Detail, bad = -3 not detail
    
    new_df = pd.DataFrame(columns=['key', 'detailsIndex', 'detailsEmpty'])
    
    fns = os.listdir('data/all/')
    
    for index, fn in enumerate(fns):
        print(index, len(fns))
        temp_df = pd.read_csv(('data/all/' + fn), compression='gzip')
        detailsIndex = list(temp_df.columns).index('Detail')
        test_detail = temp_df['Detail'].values[10]
        detailsEmpty = 0
        if type(test_detail) is float:
            detailsEmpty = 1
        new_df.loc[len(new_df.index)] = [fn.replace('.csv.gz',''), detailsIndex, detailsEmpty]
        
    new_df.to_csv("%s.csv" % "data/detailsInfo", index=False)
    
    return

def buildRawTrain():
    
    keys = ['202201150buf', '202201090tam']
    
    df = pd.read_csv("%s.csv" % "data/dataInfo")
    
    # temp = df.loc[df['key'].isin(keys)]
    
    # print(temp['detailsIndex'].values)
    
    fns = os.listdir('data/all/')
    
    # df0 = pd.read_csv("%s.csv.gz" % "data/all/202201150buf", compression='gzip') # 5
    # df1 = pd.read_csv("%s.csv.gz" % "data/all/202201090tam", compression='gzip') # 7
    
    df_list = []
    
    for index, fn in enumerate(fns):
        print(index, len(fns))
        temp_df = pd.read_csv(('data/all/' + fn), compression='gzip')
        key = fn.replace('.csv.gz','')
        info = df.loc[df['key']==key, ['detailsIndex', 'detailsEmpty']].values[0]
        cols = temp_df.columns.tolist()
        if info[1] == 0:
            if info[0] == 7:
                cols[-5:-3] = ['away_points', 'home_points']
                temp_df.columns = cols
            elif info[0] == 5:
                cols[-4:-2] = ['away_points', 'home_points']
                temp_df.columns = cols
            new_cols = [
                'Quarter', 'Time', 'Down',
                'ToGo', 'Location', 'away_points',
                'home_points', 'Detail', 'EPB',
                'EPA'
            ]
            temp_df = temp_df[new_cols]
            temp_df.columns = [col.lower() for col in temp_df.columns]
            temp_df.insert(0, 'key', fn.replace('.csv.gz',''))
            temp_df.insert(1, 'num', [i for i in range(len(temp_df.index))])
            df_list.append(temp_df)
        
    new_df = pd.concat(df_list)
    
    new_df.to_csv("%s.csv" % "data/rawTrain", index=False)
    
    return

def cleanInvalid(df: pd.DataFrame):
    kickoff_line = ''
    for index, row in df.iterrows():
        num = row['num']
        quarter = row['quarter']
        line = row['detail']
        if num == 2:
            kickoff_line = line
        if (type(quarter) is str and len(quarter) > 2) or (line == kickoff_line and num != 2):
            df.drop(index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['num'] = df.index
    return df

def cleanRawTrain():
    
    df = pd.read_csv("%s.csv" % "data/rawTrain")
    
    known_invalid_keys = ['200009030nwe', '200009100cin']
    invalid_keys = list(set(df.loc[df['quarter']=='Quarter', 'key'].values))
    invalid_keys += known_invalid_keys
    
    print(invalid_keys)
    
    # df_list = []
    
    # for index, key in enumerate(invalid_keys):
    #     print(index, len(invalid_keys))
    #     temp_df: pd.DataFrame = df.loc[df['key']==key]
    #     df_list.append(cleanInvalid(temp_df))
        
    # new_df = pd.concat(df_list)
        
    # df = df.loc[~df['key'].isin(invalid_keys)]

    # df = pd.concat([df, new_df])
    
    # df.sort_values(by=['key', 'num'], inplace=True)
    
    # df.to_csv("%s.csv" % "data/rawTrain", index=False)
    
    # get known unordered keys
    # ------------------------------------
    # all_keys = list(set(df['key'].values))
    
    # for index, key in enumerate(all_keys):
    #     nums = df.loc[df['key']==key, 'num'].values
    #     if 1 not in list(nums):
    #         print(index, len(all_keys))
    #         print(key)
    # ------------------------------------
    
    return

def addWys():
    df = pd.read_csv("%s.csv" % "data/rawTrain")
    cd = pd.read_csv("%s.csv" % "../data/gameData")
    wys = []
    for index, row in df.iterrows():
        print(index, len(df.index))
        key = row['key']
        wy = cd.loc[cd['key']==key, 'wy'].values[0]
        wys.append(wy)
    df.insert(2, 'wy', wys)
    df.to_csv("%s.csv" % "data/rawTrain", index=False)
    return

#######################

scrape()

# createInfo()

# buildRawTrain()

# merge()

# cleanRawTrain()

# addWys()

# ----------