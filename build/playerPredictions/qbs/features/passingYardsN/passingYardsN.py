import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"
DATA_PATH ="../../../../../rawData/positionData/"

def buildPassingYardsN(n):

    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "QBData"))

    cols = [("passingYards" + str(n) + "_" + str(i)) for i in range(n)]
    cols = list(source.columns) + cols
    
    df = pd.DataFrame(columns=cols)

    for index, row in source.iterrows():
        pid = row['p_id']
        key = row['key'].split("-")[1]
        sourceCols = [row['key'], row['opp_abbr'], row['wy'], pid]
        start = cd.loc[cd['game_key']==key].index.values[0]
        stats = cd.loc[
            (cd.index<start)&
            (cd['p_id']==pid)
        ]
        stats = stats.tail(n).sort_index(ascending=False)
        if stats.empty:
            stats = cd.loc[cd.index<start].tail(n).sort_index(ascending=False)
            stats = stats['passing_yards'].values
        elif len(stats.index) < n:
            dif = n - len(stats.index)
            stats = stats['passing_yards'].values
            emptyArr = np.empty(dif)
            emptyArr[:] = np.NaN
            stats = np.concatenate((stats, emptyArr))
        else:
            stats = stats['passing_yards'].values
        df.loc[len(df.index)] = sourceCols + list(stats)

    df.fillna(0, inplace=True)
    df = df.round(0)

    df.to_csv("%s.csv" % ("passingYards" + str(n)), index=False)

###############################

buildPassingYardsN(10)