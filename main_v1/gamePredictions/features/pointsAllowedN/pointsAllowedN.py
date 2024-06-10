import pandas as pd
import numpy as np
import os

def getCols(cols):
    return ['home_' + c for c in cols] + ['away_' + c for c in cols]

def getStats(n, wy, abbr, cd):
    try:
        start = cd.loc[cd['wy']==wy].index.values[0]
        stats: pd.DataFrame = cd.loc[
            ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))&
            (cd.index<start),
            ['home_abbr', 'home_points', 'away_points']
        ].tail(n)
    except IndexError:
        stats = cd.loc[
            ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr)),
            ['home_abbr', 'home_points', 'away_points']
        ].tail(n)
    pas = []
    for _, row in stats.iterrows():
        home_abbr = row['home_abbr']
        if home_abbr == abbr:
            pas.append(row['away_points'])
        else:
            pas.append(row['home_points'])
    pas.reverse()
    if len(pas) < n:
        dif = n - len(pas)
        pas += [-1 for _ in range(dif)]
    return pas

def buildPointsAllowedN(n, source: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    if 'pointsAllowedN_' + str(n) + '.csv' in os.listdir(_dir):
        print('pointsAllowedN_' + str(n) + '.csv already created.')
        return
    
    print('Creating pointsAllowedN_' + str(n) + '...')
    
    new_cols = ['pointsAllowedN_' + str(i) for i in range(n)]
    new_cols = getCols(new_cols)
    
    new_df = pd.DataFrame(columns=list(source.columns)+new_cols)
    
    for index, row in source.iterrows():
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_stats = getStats(n, wy, home_abbr, cd)
        away_stats = getStats(n, wy, away_abbr, cd)
        new_df.loc[len(new_df.index)] = list(row.values) + home_stats + away_stats
        
    new_df.to_csv("%s.csv" % (_dir + "pointsAllowedN_" + str(n)), index=False)
    
    return

def buildNewPointsAllowedN(n, source: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    print('Creating pointsAllowedN_' + str(n) + '...')
    
    new_cols = ['pointsAllowedN_' + str(i) for i in range(n)]
    new_cols = getCols(new_cols)
    
    new_df = pd.DataFrame(columns=list(source.columns)+new_cols)
    
    for index, row in source.iterrows():
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_stats = getStats(n, wy, home_abbr, cd)
        away_stats = getStats(n, wy, away_abbr, cd)
        new_df.loc[len(new_df.index)] = list(row.values) + home_stats + away_stats
    
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + "newPointsAllowedN_" + str(n)), index=False)
    
    return new_df

########################

# source = pd.read_csv("%s.csv" % "../source/source")
# cd = pd.read_csv("%s.csv" % "../../../../data/oldGameData_78")

# buildPointsAllowedN(10, source, cd, "./")