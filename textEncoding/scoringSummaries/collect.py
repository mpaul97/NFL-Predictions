import sys
sys.path.append("../../")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os
from random import randrange
import time
import regex as re
from urllib.error import HTTPError
from ordered_set import OrderedSet

from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer

from paths import DATA_PATH

def getContent(url):
    tables = pd.read_html(url)
    df = tables[1]
    df.columns = list(df.columns[:-2]) + ['away', 'home']
    return df

def getData(_dir):
    
    # make dir if does not exist
    if _dir.replace("/","") not in os.listdir("./"):
        os.mkdir(_dir)
    
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))
    
    if 'collected.csv' not in os.listdir(_dir):
        cdf = pd.DataFrame(columns=['key'])
    else:
        cdf = pd.read_csv("%s.csv" % (_dir + "collected"))
    
    df_list = []
    
    _time = 2
    
    for index, row in cd.iterrows():
        try:
            key = row['key']
            print(str(round((index / max(cd.index)), 2)) + " % : " + key)
            if key not in cdf['key'].values:
                url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
                temp_df = getContent(url)
                temp_df['key'] = key
                temp_df['num'] = [i for i in range(len(temp_df.index))]
                df_list.append(temp_df)
                cdf.loc[len(cdf.index)] = [key]
            else:
                print(key, 'already found.')
            _time = 2
        except HTTPError as err:
            print(err)
            print('Time:', _time)
            _time += 1
        time.sleep(_time)
        
    df = pd.concat(df_list)
    
    df.reset_index(drop=True, inplace=True)
    
    df.to_csv("%s.csv" % (_dir + "rawTrain"), index=False)
    
    # cdf.to_csv("%s.csv" % (_dir + "collected"), index=False)
    
    return

def convert(_dir):
    
    df = pd.read_csv("%s.csv" % (_dir + "rawTrain"))
    
    df = df[['key', 'num', 'Tm', 'Detail', 'away', 'home']]
    
    abbrs = pd.read_csv("%s.csv" % "../../teamNames/teamNames")
    
    new_df = pd.DataFrame(columns=['key', 'num', 'detail', 'info'])
    
    for index, row in df.iterrows():
        print(index, max(df.index))
        name = row['Tm']
        detail = row['Detail']
        try:
            abbr = abbrs.loc[abbrs['name'].str.contains(name), 'abbr'].values[0]
        except IndexError:
            print(name, '!! MISSING !!')
            return
        # adding info
        home_points = row['home']
        away_points = row['away']
        num = row['num']
        if num == 0:
            home_dif = home_points
            away_dif = away_points
        else:
            home_dif = home_points - df.iloc[index-1]['home']
            away_dif = away_points - df.iloc[index-1]['away']
        dif = home_dif if home_dif != 0 else away_dif
        info = abbr + '|'
        info += 'td|nex' if dif == 6 else ''
        info += 'td|ex' if dif == 7 else ''
        info += 'td|two' if dif == 8 else ''
        info += 'fg' if dif == 3 else ''
        info += 'sf' if dif == 2 else ''
        if dif == 2:
            print(detail)
        new_df.loc[len(new_df.index)] = [row['key'], num, detail, info]
        
    new_df.to_csv("%s.csv" % (_dir +"train"), index=False)
    
    return

def convertSafeties(_dir):
    
    df = pd.read_csv("%s.csv" % (_dir + "train"))
    
    sdf = df.loc[df['info'].str.contains('sf')]
    
    for index, row in sdf.iterrows():
        info = row['info']
        line = row['detail']
        if 'extra' in line:
            info_arr = info.split("|")
            info_arr[-1] = 'exrt'
            new_info = '|'.join(info_arr)
            df.at[index, 'info'] = new_info
            
    df.to_csv("%s.csv" % (_dir + "train"), index=False)
    
    return

######################

_dir = 'data/'

# getData(_dir)

# convert(_dir)

# convertSafeties(_dir)

df = pd.read_csv("%s.csv" % "train/merged/vector_merged")

keys = OrderedSet(df['key'].values)

print(keys)