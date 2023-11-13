import sys
sys.path.append("../../")
from scoringSummaries.namesRegex import getNames

import pandas as pd
import numpy as np
import regex as re
import os
from random import randrange
import time
import multiprocessing
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

pd.options.mode.chained_assignment = None

from paths import DATA_PATH

ndf = pd.read_csv("%s.csv" % "../../playerNames/playerNames")
tndf = pd.read_csv("%s.csv" % "../../teamNames/teamNames")

def showPercent(index, maxIndex):
    val = int((index/maxIndex)*100)
    if val % 10 == 0 and val > 1:
        print(str(val) + '%')
    return

# converts raw to train (adds info column (td|ex, etc.))
def rawToTrain():
    
    df = pd.read_csv("%s.csv" % "raw")
    
    df = df[['key', 'num', 'Tm', 'Detail', 'away', 'home']]
    
    new_df = pd.DataFrame(columns=['key', 'num', 'detail', 'info'])
    
    for index, row in df.iterrows():
        showPercent(index, max(df.index))
        name = row['Tm']
        detail = row['Detail']
        try:
            abbr = tndf.loc[tndf['name'].str.contains(name), 'abbr'].values[0]
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
        if 'extra' not in detail:
            info += 'sf' if dif == 2 else ''
        else:
            info += 'exrt' if dif == 2 else ''
        new_df.loc[len(new_df.index)] = [row['key'], num, detail, info]
        
    new_df.to_csv("%s.csv" % "train", index=False)
    
    return

# converts lines for given df
def infoParallelHelper(df):
    
    lines = df['detail'].values
    
    for index, line in enumerate(lines):
        if 'II' in line:
            lines[index] = line.replace('II', '')
        if 'III' in line:
            lines[index] = line.replace('III', '')
        if 'safety' not in lines[index].lower():
            names = getNames(lines[index])
            # get pids
            for name in names:
                try:
                    info = ndf.loc[
                        (ndf['name'].str.contains(name))|
                        (ndf['aka'].str.contains(name)), 
                        ['p_id', 'abbr', 'position']
                    ].values
                    pid = '|'.join(info[0])
                    if info.shape[0] > 1:
                        team_abbr = (df.iloc[index]['info']).split("|")[0]
                        for val in info:
                            if val[1] == team_abbr:
                                pid = '|'.join(val)
                except IndexError:
                    print(index, name, '!! UNKNOWN !!')
                    return
                lines[index] = lines[index].replace(name, ('|' + pid + '|'))
        else:
            lines[index] = 'Safety'
            
    df['detail'] = lines
    
    return df
        
# set info parallel
def trainToInfoParallel():
    
    df = pd.read_csv("%s.csv" % "train")
    
    df_list = []
    
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    df_split = np.array_split(df, num_partitions)
    
    if __name__ == '__main__':
        pool = multiprocessing.Pool(num_cores)
        df_list.append(pd.concat(pool.map(infoParallelHelper, df_split)))
        pool.close()
        pool.join()
    
    if __name__ == '__main__':
        if df_list:
            new_df = pd.concat(df_list)
            new_df.to_csv("%s.csv" % "info", index=False)
    
    return

############################

# rawToTrain()

# trainToInfoParallel()
    