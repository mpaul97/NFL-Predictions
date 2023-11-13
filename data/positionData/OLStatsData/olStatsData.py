import pandas as pd
import numpy as np
import os

DATA_PATH = "../../rawData/"
POSITION_PATH = DATA_PATH + "positionData/"

def zeroDivision(n, d):
    return n / d if d else 0

def getOlStarters(starters):
    return [(s.replace(":OL","")) for s in starters if 'OL' in s]

def buildStats():
    
    sdf = pd.read_csv("%s.csv" % "../simpleScrapeStarters")
    
    data = pd.concat([pd.read_csv("%s.csv" % (POSITION_PATH + "RBData")),
                      pd.read_csv("%s.csv" % (POSITION_PATH + "QBData"))])
    
    new_df = pd.DataFrame(columns=['p_id', 'game_key', 'abbr', 'wy', 'times_sacked',
                                   'yards_lost_from_sacks', 'sack_percentage',
                                   'rush_yards_per_attempt'])
    
    targetStatsQbs = ['times_sacked', 'yards_lost_from_sacks', 'sack_percentage']
    targetStatsAll = ['rush_yards_per_attempt']
    
    for index, row in sdf.iterrows():
        wy = row['wy']
        abbr = row['key'].split("-")[0]
        key = row['key'].split("-")[1]
        players = row['players'].split("|")
        starters = getOlStarters(players)
        stats_qb = data.loc[(data['game_key']==key)&(data['abbr']==abbr), targetStatsQbs]
        stats_qb.dropna(axis=0, inplace=True)
        qbLen = len(stats_qb.index)
        stats_qb = stats_qb.sum()
        stats_qb = stats_qb.apply(lambda x: zeroDivision(x, qbLen))
        stats_all = data.loc[(data['game_key']==key)&(data['abbr']==abbr), targetStatsAll]
        stats_all = stats_all.loc[~(stats_all==0).all(axis=1)]
        allLen = len(stats_all.index)
        stats_all = stats_all.sum()
        stats_all = stats_all.apply(lambda x: zeroDivision(x, allLen))
        for s in starters:
            ts = stats_qb[targetStatsQbs[0]]
            ylfs = stats_qb[targetStatsQbs[1]]
            sp = stats_qb[targetStatsQbs[2]]
            rypa = stats_all[targetStatsAll[0]]
            new_df.loc[len(new_df.index)] = [s, key, abbr, wy, ts, ylfs, sp, rypa]
        
    new_df.to_csv("%s.csv" % "OLStatsData", index=False)
    
#######################################

buildStats()