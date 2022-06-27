import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"
DATA_PATH = "../../../../rawData/"

def zeroDivision(n, d):
    return n / d if d else 0

def getWeek1Stats(abbr, year, cd):

    homeStats = cd.loc[(cd['wy'].str.contains(str(year - 1)))&(cd['home_abbr']==abbr)]
    awayStats = cd.loc[(cd['wy'].str.contains(str(year - 1)))&(cd['away_abbr']==abbr)]
    
    if homeStats.empty: # new team, use past year all teams average
        homeStats = cd.loc[(cd['wy'].str.contains(str(year - 1)))]
        
    if awayStats.empty: # new team, use past year all teams average
        awayStats = cd.loc[(cd['wy'].str.contains(str(year - 1)))]
    
    stats = []
    targets = ['points', 'total_yards', 'pass_yards', 'rush_yards']
    
    for t in targets:
        tempH = sum(list(homeStats['home_' + t]))/len(homeStats.index)
        tempA = sum(list(awayStats['away_' + t]))/len(awayStats.index)
        temp = (tempH + tempA)/2
        stats.append(temp)
        temp1H = sum(list(homeStats['away_' + t]))/len(homeStats.index)
        temp1A = sum(list(awayStats['home_' + t]))/len(awayStats.index)
        temp1 = (temp1H + temp1A)/2
        stats.append(temp1)
    
    return stats

def getStatsRest(abbr, start, year, cd):
    
    homeStats = cd.loc[(cd.index<start)&(cd['wy'].str.contains(str(year)))&(cd['home_abbr']==abbr)]
    awayStats = cd.loc[(cd.index<start)&(cd['wy'].str.contains(str(year)))&(cd['away_abbr']==abbr)]
    
    stats = []
    targets = ['points', 'total_yards', 'pass_yards', 'rush_yards']
    
    for t in targets:
        tempH = zeroDivision(sum(list(homeStats['home_' + t])), len(homeStats.index))
        tempA = zeroDivision(sum(list(awayStats['away_' + t])), len(awayStats.index))
        if tempH != 0 and tempA != 0:
            temp = (tempH + tempA)/2
        else:
            temp = tempH + tempA
        stats.append(temp)
        temp1H = zeroDivision(sum(list(homeStats['away_' + t])), len(homeStats.index))
        temp1A = zeroDivision(sum(list(awayStats['home_' + t])), len(awayStats.index))
        if temp1H != 0 and temp1A != 0:
            temp1 = (temp1H + temp1A)/2
        else:
            temp1 = temp1H + temp1A
        stats.append(temp1)
    
    return stats

def buildRankings():
    
    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    
    cols = ['ppg', 'opp_ppg', 'total_yards', 'opp_total_yards',
            'pass_yards', 'opp_pass_yards', 'rush_yards',
            'opp_rush_yards']
    
    cols = list(source.columns) + cols
    
    df = pd.DataFrame(columns=cols)
    
    for index, row in source.iterrows():
        abbr = row['key'].split("-")[0]
        key = row['key'].split("-")[1]
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        if week == 1: # averages of previous season
            stats = getWeek1Stats(abbr, year, cd)
            df.loc[-1] = [row['key'], row['opp_abbr'], wy] + stats
            df.index = df.index + 1
            df = df.sort_index(ascending=False)
        else:
            start = cd.loc[cd['wy']==wy].index.values[0]
            stats = getStatsRest(abbr, start, year, cd)
            df.loc[-1] = [row['key'], row['opp_abbr'], wy] + stats
            df.index = df.index + 1
            df = df.sort_index(ascending=False)
    
    df.to_csv("seasonRankings.csv", index=False)
    
#################################

buildRankings()