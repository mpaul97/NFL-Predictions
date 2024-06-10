import pandas as pd
import numpy as np
import os

def zeroDivision(num, den):
    try:
        val = num/den
    except ZeroDivisionError:
        val = 0
    return val

def getCols(cols):
    return ['home_' + c for c in cols] + ['away_' + c for c in cols]

def getData(abbr, stats: pd.DataFrame):
    stats = stats.loc[(stats['home_abbr']==abbr)|(stats['away_abbr']==abbr), 'winning_abbr'].values
    wins = list(stats).count(abbr)
    loses = len(stats) - wins
    wlp = zeroDivision(wins, len(stats)) * 100
    return [wins, loses, wlp]

def buildSeasonInfo(source: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    if 'seasonInfo.csv' in os.listdir(_dir):
        print('seasonInfo.csv already built.')
        return
    
    print('Creating seasonInfo...')
    
    cols = ['week', 'season', 'seasonWins', 'seasonLoses', 'seasonWLP', 'isHome']
    cols = getCols(cols)
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    
    for index, row in source.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        start = cd.loc[cd['wy']==wy].index.values[0]
        stats = cd.loc[
            (cd.index<start)&
            (cd['wy'].str.contains(str(year)))
        ]
        home_data = getData(home_abbr, stats)
        away_data = getData(away_abbr, stats)
        home_data = [week, year] + home_data + [1]
        away_data = [week, year] + away_data + [0]
        new_df.loc[len(new_df.index)] = list(row.values) + home_data + away_data
        
    new_df.to_csv("%s.csv" % (_dir + "seasonInfo"), index=False)
    
    return

def buildNewSeasonInfo(source: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    print('Creating newSeasonInfo...')
    
    cols = ['week', 'season', 'seasonWins', 'seasonLoses', 'seasonWLP', 'isHome']
    cols = getCols(cols)
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    
    for index, row in source.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        try:
            start = cd.loc[cd['wy']==wy].index.values[0]
            stats = cd.loc[
                (cd.index<start)&
                (cd['wy'].str.contains(str(year)))
            ]
        except IndexError:
            stats = cd.loc[
                (cd['wy'].str.contains(str(year)))
            ]
        home_data = getData(home_abbr, stats)
        away_data = getData(away_abbr, stats)
        home_data = [week, year] + home_data + [1]
        away_data = [week, year] + away_data + [0]
        new_df.loc[len(new_df.index)] = list(row.values) + home_data + away_data
        
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + 'newSeasonInfo'), index=False)
    
    return new_df

# source = pd.read_csv("%s.csv" % "../source/source")
# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")

# buildSeasonInfo(source, cd, './')
