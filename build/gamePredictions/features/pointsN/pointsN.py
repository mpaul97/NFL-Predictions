from turtle import home
import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"
DATA_PATH = "../../../../rawData/"

def buildPointsN(n):
    
    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    
    cols = [("points" + str(n) + "_" + str(i)) for i in range(n)]
    cols = list(source.columns) + cols
    
    df = pd.DataFrame(columns=cols)
    
    for index, row in source.iterrows():
        abbr = row['key'].split("-")[0]
        key = row['key'].split("-")[1]
        sourceCols = [row['key'], row['opp_abbr'], row['wy']]
        start = cd.loc[cd['key']==key].index[0]
        stats = cd.loc[
            (cd.index<start)&
            ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))
        ]
        stats = stats.tail(n)
        if stats.empty:
            temp = cd.loc[cd.index<start].tail(int(n/2))
            homePoints = temp['home_points']
            awayPoints = temp['away_points']
            points = pd.concat([homePoints, awayPoints]).sort_index(ascending=False).values
        elif len(stats.index) < n:
            dif = n - len(stats.index)
            homePoints = stats.loc[stats['home_abbr']==abbr, 'home_points']
            awayPoints = stats.loc[stats['away_abbr']==abbr, 'away_points']
            points = pd.concat([homePoints, awayPoints]).sort_index(ascending=False).values
            emptyArr = np.empty(dif)
            emptyArr[:] = np.NaN
            points = np.concatenate((points, emptyArr))
        else:
            homePoints = stats.loc[stats['home_abbr']==abbr, 'home_points']
            awayPoints = stats.loc[stats['away_abbr']==abbr, 'away_points']
            points = pd.concat([homePoints, awayPoints]).sort_index(ascending=False).values
        df.loc[len(df.index)] = sourceCols + list(points)
            
    df.fillna(df.mean(), inplace=True)
    df = df.round(0)
    
    df.to_csv("%s.csv" % ("points" + str(n)), index=False)
        
######################################

buildPointsN(10)