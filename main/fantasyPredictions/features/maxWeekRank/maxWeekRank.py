import pandas as pd
import numpy as np
import os

def buildMaxWeekRank(source: pd.DataFrame, df: pd.DataFrame, _dir):
    if 'maxWeekRank.csv' in os.listdir(_dir):
        print('maxWeekRank already created.')
        return
    print('Creating maxWeekRank...')
    all_ranks = []
    for index, row in source.iterrows():
        wy = row['wy']
        position = row['position']
        ranks = df.loc[(df['wy']==wy)&(df['position']==position), 'week_rank'].values
        max_rank = max(ranks)
        all_ranks.append(max_rank)
    source['maxWeekRank'] = all_ranks
    source.to_csv("%s.csv" % (_dir + "maxWeekRank"), index=False)
    return

def buildNewMaxWeekRank(source: pd.DataFrame, _dir):
    print('Creating maxWeekRank...')
    all_ranks = []
    for index, row in source.iterrows():
        position = row['position']
        ranks = source.loc[source['position']==position, 'position'].values
        max_rank = len(ranks)
        all_ranks.append(max_rank)
    source['maxWeekRank'] = all_ranks
    return source

##########################

# source = pd.read_csv("%s.csv" % "../source/source")
# df = pd.read_csv("%s.csv" % "../../../../data/fantasyData")

# buildMaxWeekRank(source, df, './')

# source = pd.read_csv("%s.csv" % "../source/new_source")

# buildNewMaxWeekRank(source, './')