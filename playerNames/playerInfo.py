import pandas as pd
import numpy as np
import os
import urllib.request
from urllib.error import HTTPError
import requests
from bs4 import BeautifulSoup
import regex as re
import time
import math
from ordered_set import OrderedSet

pd.options.mode.chained_assignment = None

def mostCommon(List):
    return max(set(List), key = List.index)

def getCounts(List):
    counts = { l:List.count(l) for l in List }
    return counts

def getNewYears(df):
    curr_years = df['year'].values
    new_years = []
    last_valid_year = 0
    for index, year in enumerate(curr_years):
        if type(year) is float and math.isnan(year):
            if index != 0:
                new_years.append(last_valid_year)
        else:
            if (type(year) is str or type(year) is object) and '*' in year:
                year = year.replace('*', '')
            if (type(year) is str or type(year) is object) and '+' in year:
                year = year.replace('+', '')
            last_valid_year = year
            new_years.append(year)
    return new_years

def getContent(pid):
    
    url_c = pid[0].upper()
    url = 'https://www.pro-football-reference.com/players/' + url_c + '/' + pid + '.htm'
    
    # fp = urllib.request.urlopen(url)
    # mybytes = fp.read()
    # mystr = mybytes.decode("utf8", errors='ignore')
    # fp.close()
    
    tables = pd.read_html(url)
    
    df_list = []
    
    for t in tables:
        if type(t.columns[0]) is tuple or len(tables) == 1 or (type(t.columns[0]) is str and 'Year' in t.columns):
            temp_df = t[t.columns[:4]]
            temp_df.columns = ['year', 'age', 'team', 'position']
            temp_df['year'] = getNewYears(temp_df)
            temp_df['year'] = pd.to_numeric(temp_df['year'], errors='coerce')
            temp_df.dropna(subset=['year'], inplace=True)
            df_list.append(temp_df)
            
    df = pd.concat(df_list)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=['year'], inplace=True)
    df.drop(columns=['age'], inplace=True)
    df.dropna(inplace=True)
    
    position = mostCommon([pos for pos in df['position'].values if 'Missed season' not in pos])
    
    info = []
    
    for index, row in df.iterrows():
        year = row['year']
        team = row['team']
        info.append(team + ',' + str(int(year)))
    
    info = list(OrderedSet(info))
    
    return position, '|'.join(info)

def build():
    
    df = pd.concat([
        pd.read_csv("%s.csv" % "playerNames"),
        pd.read_csv("%s.csv" % "finalPlayerNames_pbp")
    ])
    
    if 'playerInfo.csv' not in os.listdir("./"):
        new_df = pd.DataFrame(columns=['p_id', 'name', 'position', 'info'])
    else:
        new_df = pd.read_csv("%s.csv" % "playerInfo")
        drop_indexes = [i for i in range(0, len(new_df.index))]
        df.drop(drop_indexes, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    for index, row in df.iterrows():
        print(index, len(df.index))
        name = row['name']
        pid = row['p_id']
        print(pid + '\n')
        try:
            position, info = getContent(pid)
            new_df.loc[len(new_df.index)] = [pid, name, position, info]
            time.sleep(2)
        except (HTTPError, ValueError) as error:
            print(error)
            break
    
    new_df.to_csv("%s.csv" % "playerInfo", index=False)
    
    return

def buildFinalPbpNames():
    
    pn = pd.read_csv("%s.csv" % "playerNames")
    df = pd.read_csv("%s.csv" % "finalPlayerNames_pbp")

    new_df = pd.DataFrame(columns=['p_id', 'name', 'position', 'info'])
    
    index = 0
    _time = 5
    
    while index != len(df.index) - 1:
        print(index, len(df.index))
        row = df.iloc[index]
        pid = row['p_id']
        if pid not in pn['p_id'].values:
            name = row['name']
            print(pid + '\n')
            try:
                position, info = getContent(pid)
                new_df.loc[len(new_df.index)] = [pid, name, position, info]
                time.sleep(_time)
                index += 1
            except ValueError as error:
                print('ValueError:', pid, error)
                new_df.loc[len(new_df.index)] = [pid, name, 'UNK', 'UNK']
                index += 1
            except HTTPError as error:
                if error.code != 429:
                    print('HttpError normal:', pid, error)
                    new_df.loc[len(new_df.index)] = [pid, name, 'UNK', 'UNK']
                    index += 1
                else:
                    print('Time:', _time, error)
                    _time += 1
                    time.sleep(_time)
        else:
            index += 1
    
    new_df.to_csv("%s.csv" % "finalPbpNames-playerInfo", index=False)
    
    return

def addAKA():
    
    df = pd.read_csv("%s.csv" % "playerInfo")
    pn = pd.read_csv("%s.csv" % "playerNames")
    
    pn = pn[['p_id', 'aka']]
    
    print(df.shape, pn.shape)
    
    new_df = df.merge(pn, on=['p_id'], how='left')
    
    new_df.drop_duplicates(inplace=True)
    
    new_df.to_csv("%s.csv" % "playerInfo", index=False)
    
    return

def mergePbpNames():
    
    pn = pd.read_csv("%s.csv" % "playerNames")
    df = pd.read_csv("%s.csv" % "finalPbpNames-playerInfo")
    
    new_df = pd.concat([pn, df])
    
    print(new_df.loc[new_df['name'].str.contains('Adam Lingner')])
    
    new_df.drop_duplicates(inplace=True)
    
    print(new_df.loc[new_df['name'].str.contains('Adam Lingner')])
    
    new_df.to_csv("%s.csv" % "playerInfo", index=False)
    
    return

def clean():
    
    df = pd.read_csv("%s.csv" % "playerInfo")
    pn = pd.read_csv("%s.csv" % "playerNames")
    
    new_df = pd.concat([df, pn])
    
    print(new_df.shape)
    new_df.drop_duplicates(inplace=True)
    print(new_df.shape)
    
    new_df = new_df[new_df['info'].notna()]
    print(new_df.shape)
    
    new_df.to_csv("%s.csv" % "playerInfo", index=False)
    
    return

def rebuild():
    
    pn = pd.read_csv("%s.csv" % "playerNames")
    df = pd.read_csv("%s.csv" % "playerInfo")

    new_df = pd.DataFrame(columns=['p_id', 'name', 'position', 'info'])
    
    index = 0
    _time = 5
    
    while index != len(pn.index) - 1:
        print(index, len(pn.index))
        row = pn.iloc[index]
        pid = row['p_id']
        if pid not in df['p_id'].values:
            name = row['name']
            print(pid + '\n')
            try:
                position, info = getContent(pid)
                new_df.loc[len(new_df.index)] = [pid, name, position, info]
                time.sleep(_time)
                index += 1
            except ValueError as error:
                print('ValueError:', pid, error)
                new_df.loc[len(new_df.index)] = [pid, name, 'UNK', 'UNK']
                index += 1
            except HTTPError as error:
                if error.code != 429:
                    print('HttpError normal:', pid, error)
                    new_df.loc[len(new_df.index)] = [pid, name, 'UNK', 'UNK']
                    index += 1
                else:
                    print('Time:', _time, error)
                    _time += 1
                    time.sleep(_time)
        else:
            index += 1
    
    new_df.to_csv("%s.csv" % "playerInfo-1", index=False)
    
    return

def mergeRebuild():
    
    df = pd.read_csv("%s.csv" % "playerInfo")
    df1 = pd.read_csv("%s.csv" % "playerInfo-1")
    
    pn = pd.read_csv("%s.csv" % "playerNames")
    pn = pn[['p_id', 'aka']]
    
    df.drop(columns=['aka'], inplace=True)
    
    new_df = pd.concat([df, df1])
    
    new_df.drop_duplicates(inplace=True)
    
    new_df = new_df.merge(pn, on=['p_id'])
    
    # store duplicates and edit
    # dup_df = pd.concat(g for _, g in new_df.groupby("p_id") if len(g) > 1)
    
    # dup_df.to_csv("%s.csv" % "dupPlayerInfo")
    
    dup_df = pd.read_csv("%s.csv" % "dupPlayerInfo")
    dup_df.drop(columns=['idx'], inplace=True)
    
    dup_pids = dup_df['p_id'].values
    
    new_df.drop(new_df.loc[new_df['p_id'].isin(dup_pids)].index, axis=0, inplace=True)
    
    new_df = pd.concat([new_df, dup_df])
    
    new_df.sort_values(by=['name'], inplace=True)
    
    new_df.to_csv("%s.csv" % "finalPlayerInfo", index=False)
    
    return

def fixMissedSeason():
    
    df = pd.read_csv("%s.csv" % "finalPlayerInfo")
    
    # positions = list(set(df['position'].values))
    # positions.sort()
    
    # for pos in positions:
    #     print(pos)
    
    mdf = df.loc[df['position'].str.contains('Missed season')]
    
    for index, row in mdf.iterrows():
        print(index)
        pid = row['p_id']
        pos, info = getContent(pid)
        df.at[index, 'position'] = pos
        df.at[index, 'info'] = info
        time.sleep(5)
    
    df.to_csv("%s.csv" % "finalPLayerInfo", index=False)
    
    return

def savePositions():
    
    df = pd.read_csv("%s.csv" % "finalPlayerInfo")
    
    positions = getCounts(list(df['position'].values))
    
    pos_df = pd.DataFrame.from_dict(positions, orient='index')
    pos_df.insert(0, 'position', pos_df.index)
    pos_df.reset_index(drop=True, inplace=True)
    pos_df.columns = ['position', 'count']
    pos_df.sort_values(by=['count'], ascending=False, inplace=True)
    
    pos_df.to_csv("%s.csv" % "positionsFinalPlayerInfo", index=False)
    
    return

def simplifyPositions():
    
    df = pd.read_csv("%s.csv" % "finalPlayerInfo")
    pos_df = pd.read_csv("%s.csv" % "positionsFinalPlayerInfo")
    
    for index, row in df.iterrows():
        print(index, len(df.index))
        position = row['position']
        s_position = pos_df.loc[pos_df['position']==position, 'simplePosition'].values[0]
        df.at[index, 'position'] = s_position
        
    df.to_csv("%s.csv" % "finalPlayerInfo", index=False)
    
    return

def getPositions():
    pos_df = pd.read_csv("%s.csv" % "positionsFinalPlayerInfo")
    positions = list(set(pos_df['simplePosition'].values))
    print(positions)
    return

def updateFinalPlayerInfo():
    data_dir = "../data/"
    df = pd.read_csv("%s.csv" % (data_dir + "allPids"))
    all_pids = '|'.join(df['pids'])
    new_df = pd.DataFrame()
    new_df['p_id'] = all_pids.split("|")
    new_df['position'] = new_df['p_id'].apply(lambda x: x.split(":")[1])
    new_df['p_id'] = new_df['p_id'].apply(lambda x: x.split(":")[0])
    new_df = new_df.loc[new_df['position']=='UNK']
    new_df.drop_duplicates(inplace=True)
    fp = pd.read_csv("%s.csv" % "finalPlayerInfo")
    pos_df = pd.read_csv("%s.csv" % "positionsFinalPlayerInfo")
    for index, pid in enumerate(new_df['p_id'].values):
        print(round((index/len(new_df.index)*100), 2))
        position, info = getContent(pid)
        s_pos = pos_df.loc[pos_df['position']==position, 'simplePosition'].values[0]
        fp.loc[len(fp.index)] = [pid, 'UNK', s_pos, info, np.nan]
        time.sleep(2)
    fp.to_csv("%s.csv" % "finalPlayerInfo", index=False)
    return

##########################

# build()

# buildFinalPbpNames()

# mergePbpNames()

# clean()

# rebuild playerInfo, most INFO is missing !!!!!

# rebuild()

# mergeRebuild()

# fixMissedSeason()

# savePositions()

# simplifyPositions()

# getPositions()

# updateFinalPlayerInfo()

