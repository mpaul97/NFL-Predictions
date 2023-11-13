import pandas as pd
import numpy as np
import os

pd.options.mode.chained_assignment = None

def getIsHalfGame(stats):
    stats['ap'] = stats['attempted_passes'].apply(lambda x: x/sum(stats['attempted_passes'].values))
    stats = stats.loc[stats['ap']>0.3]
    isHalfGame = 1 if len(stats.index) > 1 else 0
    return isHalfGame

def buildQbHalfGame(source: pd.DataFrame, qdf: pd.DataFrame, _dir):
    
    if 'qbHalfGame.csv' in os.listdir(_dir):
        print('qbHalfGame.csv already created.')
        return
    
    print('Creating qbHalfGame...')
    
    new_df = pd.DataFrame(columns=list(source.columns)+['home_isQbHalfGame', 'away_isQbHalfGame'])
    
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_stats = qdf.loc[(qdf['game_key']==key)&(qdf['abbr']==home_abbr)]
        away_stats = qdf.loc[(qdf['game_key']==key)&(qdf['abbr']==away_abbr)]
        if len(home_stats.index) > 1:
            home_isHalfGame = getIsHalfGame(home_stats)
        else:
            home_isHalfGame = 0
        if len(away_stats.index) > 1:
            away_isHalfGame = getIsHalfGame(away_stats)
        else:
            away_isHalfGame = 0
        new_df.loc[len(new_df.index)] = list(row.values) + [home_isHalfGame, away_isHalfGame]
        
    new_df.to_csv("%s.csv" % (_dir + "qbHalfGame"), index=False)
    
    return

def buildNewQbHalfGame(source: pd.DataFrame, _dir):
    print('Creating qbHalfGame...')
    new_df = source.copy()
    new_df['home_isQbHalfGame'] = 0
    new_df['away_isQbHalfGame'] = 0
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + 'newQbHalfGame'), index=False)
    return new_df

########################

# source = pd.read_csv("%s.csv" % "../source/source")
# qdf = pd.read_csv("%s.csv" % "../../../../data/positionData/QBData")

# buildQbHalfGame(source, qdf, './')

# source = pd.read_csv("%s.csv" % "../source/new_source")

# buildNewQbHalfGame(source, './')

# _dict = {'A': ['a', 'b'], 'B': [20, 4]}

# df = pd.DataFrame(_dict)

# df['perc'] = df['B'].apply(lambda x: x/sum(df['B'].values))

# print(df)