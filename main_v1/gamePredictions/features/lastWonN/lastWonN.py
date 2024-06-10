import pandas as pd
import numpy as np
import os

def getCols(cols):
    return ['home_' + c for c in cols] + ['away_' + c for c in cols]

def getWins(n, abbr, wy, cd: pd.DataFrame):
    try:
        start = cd.loc[cd['wy']==wy].index.values[0]
        stats = cd.loc[
            ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))&
            (cd.index<start),
            'winning_abbr'
        ].tail(n)
    except IndexError:
        stats = cd.loc[
            ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr)),
            'winning_abbr'
        ].tail(n)
    stats = [(1 if s == abbr else 0) for s in stats]
    stats.reverse()
    if len(stats) < n:
        dif = n - len(stats)
        stats += [-1 for _ in range(dif)]
    return stats

def buildLastWonN(n, source: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    if 'lastWonN_' + str(n) + '.csv' in os.listdir(_dir):
        print('lastWonN_' + str(n) + '.csv already created.')
        new_df = pd.read_csv("%s.csv" % (_dir + "lastWonN_" + str(n)))
        new_df.to_csv("%s.csv" % (_dir + 'lastWonN_' + str(n)), index=False)
        return
    
    print('Creating lastWonN_' + str(n) + '...')
    
    new_cols = ['lastWonN_' + str(i) for i in range(n)]
    new_cols = getCols(new_cols)
    
    new_df = pd.DataFrame(columns=list(source.columns)+new_cols)
    
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_wins = getWins(n, home_abbr, wy, cd)
        away_wins = getWins(n, away_abbr, wy, cd)
        new_df.loc[len(new_df.index)] = list(row.values) + home_wins + away_wins
        
    new_df.to_csv("%s.csv" % (_dir + 'lastWonN_' + str(n)), index=False)
    
    return

def buildNewLastWonN(n, source: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    print('Creating newLastWonN_' + str(n) + '...')
    
    new_cols = ['lastWonN_' + str(i) for i in range(n)]
    new_cols = getCols(new_cols)
    
    new_df = pd.DataFrame(columns=list(source.columns)+new_cols)
    
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_wins = getWins(n, home_abbr, wy, cd)
        away_wins = getWins(n, away_abbr, wy, cd)
        new_df.loc[len(new_df.index)] = list(row.values) + home_wins + away_wins
        
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + 'newLastWonN_' + str(n)), index=False)
        
    return new_df

#########################

# source = pd.read_csv("%s.csv" % "../source/source")
# cd = pd.read_csv("%s.csv" % "../../../../data/oldGameData_78")

# buildLastWonN(10, source, cd, './')