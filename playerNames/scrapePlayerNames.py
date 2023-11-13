import pandas as pd
import numpy as np
import urllib.request
import os

DATA_PATH = "../rawData/"
NEWDATA_PATH = "../2022data/"

def getContent(url):
    
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()

    mystr = mybytes.decode("utf8", errors='ignore')
    fp.close()

    temp = mystr[mystr.index('id="info"'):mystr.index('</h1>')]
    
    temp = temp[temp.index('<span>'):temp.index('</span>')]
    
    name = temp.replace('<span>', '')
    
    return name

def build():
    
    df = pd.read_csv("%s.csv" % (DATA_PATH + "playerData_78-21W20"))
    
    pids = df['p_id'].values

    pids = list(set(pids))
    
    new_df = pd.DataFrame(columns=['p_id', 'name'])
    
    count = 0
    for pid in pids:
        urlKey = pid[0].upper()
        url = 'https://www.pro-football-reference.com/players/' + urlKey + '/' + pid + '.htm'
        name = getContent(url)
        new_df.loc[len(new_df.index)] = [pid, name]
        # get progress
        print(round(count / len(pids), 2)*100)
        count += 1
        
    new_df.to_csv("%s.csv" % "playerNames", index=False)
    
def append():
  
    dfn = pd.read_csv("%s.csv" % "playerNames")
    df = pd.read_csv("%s.csv" % (NEWDATA_PATH + "finalSeasonRanks")) # should contain all new pids
    
    foundPids = list(set(dfn['p_id'].values))
    
    player_cols = [col for col in df.columns if col != 'abbr' and col != 'year']
    
    new_df = pd.DataFrame(columns=['p_id', 'name'])
    
    for index, row in df.iterrows():
        for col in player_cols:
            pids = row[col].split("|")
            for pid in pids:
                if pid not in foundPids:
                    urlKey = pid[0].upper()
                    url = 'https://www.pro-football-reference.com/players/' + urlKey + '/' + pid + '.htm'
                    name = getContent(url)
                    if pid not in list(set(new_df['p_id'].values)):
                        print(name)
                        new_df.loc[len(new_df.index)] = [pid, name]
                    
    dfn = pd.concat([dfn, new_df])
    
    dfn.to_csv("%s.csv" % "playerNames", index=False)
    
    return

def buildOffNames():

    df = pd.read_csv("%s.csv" % (NEWDATA_PATH + "finalSeasonRanks")) # should contain all new pids
    
    player_cols = [col for col in df.columns if col != 'abbr' and col != 'year']
    player_cols = player_cols[:4]
    
    new_df = pd.DataFrame(columns=['p_id', 'name'])
    
    for index, row in df.iterrows():
        for col in player_cols:
            pids = row[col].split("|")
            for pid in pids:
                urlKey = pid[0].upper()
                url = 'https://www.pro-football-reference.com/players/' + urlKey + '/' + pid + '.htm'
                name = getContent(url)
                if pid not in list(set(new_df['p_id'].values)):
                    print(name)
                    new_df.loc[len(new_df.index)] = [pid, name]
    
    new_df.to_csv("%s.csv" % "playerNamesOff", index=False)
    
    return
    
###############################################

# url = 'https://www.pro-football-reference.com/players/E/EvanDa02.htm'

# name = getContent(url)

# build()

# append()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!  edited - missing players added and aka added !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
buildOffNames()