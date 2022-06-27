import pandas as pd
import numpy as np

SOURCE_PATH = "../joining/"
TARGET_PATH = "../../targets/"
DATA_PATH = "../../../../rawData/"

def replaceColsFast(temp, isHome, N):
    if isHome:
        col_names = list(temp.columns)
        for name in col_names:
            new_col_name = name.replace('home_', 'time_' + str(N) + '_').replace('away_', 'time_' + str(N) + '_opp_')
            temp = temp.rename(columns={name: new_col_name})
            # temp = temp.rename(index={temp.last_valid_index(): abbr})
    else:
        col_names = list(temp.columns)
        for name in col_names:
            new_col_name = name.replace('away_', 'time_' + str(N) + '_').replace('home_', 'time_' + str(N) + '_opp_')
            temp = temp.rename(columns={name: new_col_name})
            # temp = temp.rename(index={temp.last_valid_index(): abbr})
    return temp

def buildTimeN(n):
    
    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    info = pd.read_csv("%s.csv" % (SOURCE_PATH + "sourceInfo"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    
    drop_cols = ['attendance', 'stadium_id', 'lineHit', 'month', 
                 'ouHit', 'surface']
    
    cd.drop(columns=drop_cols, inplace=True)
    
    keys, statsList = [], []
    
    for index, row in source.iterrows():
        keys.append(row['key'])
        abbr = row['key'].split("-")[0]
        key = row['key'].split("-")[1]
        time = info.loc[info['key']==row['key'], 'time'].values[0]
        start = cd.loc[cd['key']==key].index[0]
        stats = cd.loc[(cd.index<start)&((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))&(cd['time']==time)]
        stats = stats.tail(n)
        if stats.empty:
            statsH = cd.loc[cd.index<start].tail(20)
            statsH = replaceColsFast(statsH, True, n)
            statsA = cd.loc[cd.index<start].tail(20)
            statsA = replaceColsFast(statsA, False, n)
            stats = pd.concat([statsH, statsA])
        else:
            homeStats = stats.loc[stats['home_abbr']==abbr]
            homeStats = replaceColsFast(homeStats, True, n)
            awayStats = stats.loc[stats['away_abbr']==abbr]
            awayStats = replaceColsFast(awayStats, False, n)
            stats = pd.concat([homeStats, awayStats])
        stats.drop(columns=['time'], inplace=True)
        num = len(stats.index)
        stats = stats.sum(numeric_only=True).to_frame().transpose()
        stats = stats.apply(lambda x: x/num)
        statsList.append(stats)
        
    new_df = pd.concat(statsList)
    new_df.insert(0, 'key', keys)
    
    source = source.merge(new_df)
    
    source.to_csv("%s.csv" % ("time" + str(n)), index=False)
    
#########################################

buildTimeN(5)