import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"
DATA_PATH ="../../../../../rawData/positionData/"

def renameCols(temp, n):
    cols = list(temp.columns)
    for col in cols:
        new_col = "last" + str(n) + "_" + col
        temp = temp.rename(columns={col: new_col})
    return temp

def buildLastN(n):

    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    df = pd.read_csv("%s.csv" % (DATA_PATH + "QBData"))

    statsList = []

    for index, row in source.iterrows():
        pid = row['p_id']
        key = row['key'].split("-")[1]
        if pid != 'UNK':
            start = df.loc[(df['p_id']==pid)&(df['game_key']==key)].index.values[0]
            stats = df.loc[(df.index<start)&(df['p_id']==pid)]
            stats = stats.tail(n)
        else:
            stats = pd.DataFrame()
            stats.insert(0, 'key', row['key'])
        if stats.empty:
            stats = pd.concat([df.loc[df.index==start], df.loc[df.index<start].tail(n-1)])
        num = len(stats)
        stats = stats.sum(numeric_only=True).to_frame().transpose()
        stats = stats.apply(lambda x: x/num)
        stats = renameCols(stats, n)
        statsList.append(stats)
            
    new_df = pd.concat(statsList)
    new_df.insert(0, 'key', list(source['key']))

    source = source.merge(new_df)

    source.to_csv("%s.csv" % ("last" + str(n)), index=False)

###################################

buildLastN(1)
