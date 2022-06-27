import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"
DATA_PATH ="../../../../../rawData/positionData/"

def renameCols(temp):
    cols = list(temp.columns)
    for col in cols:
        new_col = "seasonAvg_" + col
        temp = temp.rename(columns={col: new_col})
    return temp

def buildSeasonAvg():

    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    df = pd.read_csv("%s.csv" % (DATA_PATH + "QBData"))

    statsList = []

    for index, row in source.iterrows():
        pid = row['p_id']
        key = row['key'].split("-")[1]
        week = int(row['wy'].split(" | ")[0])
        year = int(row['wy'].split(" | ")[1])
        if pid != 'UNK':
            if week == 1:
                stats = df.loc[(df['p_id']==pid)&(df['wy'].str.contains(str(year-1)))]
                if stats.empty:
                    stats = df.loc[df['wy'].str.contains(str(year-1))].tail(20)
            else:
                start = df.loc[(df['p_id']==pid)&(df['game_key']==key)].index.values[0]
                stats = df.loc[(df.index<start)&(df['p_id']==pid)&(df['wy'].str.contains(str(year)))]
        else:
            start = df.loc[df['wy']==row['wy']].index.values[0]
            stats = df.loc[df.index<start].tail(20)
        num = len(stats)
        stats = stats.sum(numeric_only=True).to_frame().transpose()
        stats = stats.apply(lambda x: x/num)
        stats = renameCols(stats)
        statsList.append(stats)
            
    new_df = pd.concat(statsList)
    new_df.insert(0, 'key', list(source['key']))

    new_df.fillna(new_df.mean(), inplace=True)
    new_df = new_df.round(3)

    source = source.merge(new_df)

    source.to_csv("%s.csv" % "seasonAvg", index=False)

###################################

buildSeasonAvg()
