import pandas as pd
import numpy as np
import os

DATA_PATH = "../rawData/"
POSITION_DATA = DATA_PATH + "positionData/"
SCRAPE_PATH = "scrapeStarters/"

def getQb1(starters, key, abbr):
    df = pd.read_csv("%s.csv" % (POSITION_DATA + 'QBData'))
    qbs = [p.split(":")[0] for p in starters if ':QB' in p]
    if len(qbs) == 0:
        return 'UNK'
    elif len(qbs) == 1:
        return qbs[0]
    elif len(qbs) == 2:
        stats = df.loc[(df['game_key']==key)&(df['abbr']==abbr)]
        stats = stats.sort_values(by=['attempted_passes'], ascending=False)
        return stats['p_id'].values[0]    

def buildQb():
    
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    ss = pd.read_csv("%s.csv" % ('mergedScrapeStarters'))
    
    keys, wys, qb1s = [], [], []
    
    for index, row in cd.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_key = home_abbr + "-" + key
        away_key = away_abbr + "-" + key
        keys.append(home_key)
        keys.append(away_key)
        home_starters = ss.loc[ss['key']==home_key, 'merged'].values[0].split("|")
        away_starters = ss.loc[ss['key']==away_key, 'merged'].values[0].split("|")
        home_qb1 = getQb1(home_starters, key, home_abbr)
        away_qb1 = getQb1(away_starters, key, away_abbr)
        qb1s.append(home_qb1)
        qb1s.append(away_qb1)
        wys.append(row['wy'])
        wys.append(row['wy'])
        
    df = pd.DataFrame()
    df['key'] = keys
    df['wy'] = wys
    df['qb1'] = qb1s
    
    df.to_csv("%s.csv" % ("scrapePosition_qb1"), index=False)
        
def getRb1(starters, key, abbr):
    df = pd.read_csv("%s.csv" % (POSITION_DATA + 'QBData'))
    qbs = [p.split(":")[0] for p in starters if ':QB' in p]
    if len(qbs) == 0:
        return 'UNK'
    elif len(qbs) == 1:
        return qbs[0]
    elif len(qbs) == 2:
        stats = df.loc[(df['game_key']==key)&(df['abbr']==abbr)]
        stats = stats.sort_values(by=['attempted_passes'], ascending=False)
        return stats['p_id'].values[0]    

def buildRb():
    
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    ss = pd.read_csv("%s.csv" % ('mergedScrapeStarters'))
    
    keys, wys, qb1s = [], [], []
    
    for index, row in cd.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_key = home_abbr + "-" + key
        away_key = away_abbr + "-" + key
        keys.append(home_key)
        keys.append(away_key)
        home_starters = ss.loc[ss['key']==home_key, 'merged'].values[0].split("|")
        away_starters = ss.loc[ss['key']==away_key, 'merged'].values[0].split("|")
        home_qb1 = getQb1(home_starters, key, home_abbr)
        away_qb1 = getQb1(away_starters, key, away_abbr)
        qb1s.append(home_qb1)
        qb1s.append(away_qb1)
        wys.append(row['wy'])
        wys.append(row['wy'])
        
    df = pd.DataFrame()
    df['key'] = keys
    df['wy'] = wys
    df['qb1'] = qb1s
    
    df.to_csv("%s.csv" % ("scrapePosition_qb1"), index=False)

######################################

buildQb()