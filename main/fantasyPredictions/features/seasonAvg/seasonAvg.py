import pandas as pd
import numpy as np
import os
import multiprocessing
import regex as re
from functools import partial
import time

def func(source: pd.DataFrame, df: pd.DataFrame):
    # ROOT_DIR = os.path.abspath(__file__)
    # n_idx = ROOT_DIR.index('NFLPredictions3')
    # ROOT_DIR = ROOT_DIR[:n_idx-1]
    # df = pd.read_csv("%s.csv" % (ROOT_DIR + "/NFLPredictions3/data/fantasyData"))
    all_avgs = []
    for index, row in source.iterrows():
        pid = row['p_id']
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        start = df.loc[df['wy']==wy].index.values[0]
        if week != 1:
            stats = df.loc[
                (df['p_id']==pid)&
                (df['wy'].str.contains(str(year)))&
                (df.index<start),
                'points'
            ].values
        else:
            stats = df.loc[
                (df['p_id']==pid)&
                (df['wy'].str.contains(str(year-1))),
                'points'
            ].values
        if len(stats) > 1:
            s_avg = np.mean(stats)
        else:
            s_avg = np.NaN
        all_avgs.append(s_avg)
    source['seasonAvg'] = all_avgs
    return source

def buildSeasonAvg(source: pd.DataFrame, df: pd.DataFrame, _dir):
    if 'seasonAvg.csv' in os.listdir(_dir):
        print('seasonAvg already created.')
        return
    print('Creating seasonAvg...')
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    source_split = np.array_split(source, num_partitions)
    df_list = []
    if __name__ == 'fantasyPredictions.features.seasonAvg.seasonAvg':
        start = time.time()
        pool = multiprocessing.Pool(num_cores)
        all_dfs = pd.concat(pool.map(partial(func, df=df), source_split))
        df_list.append(all_dfs)
        pool.close()
        pool.join()
        new_df = pd.concat(df_list)
        new_df.fillna(0, inplace=True)
        new_df.to_csv("%s.csv" % (_dir + "seasonAvg"), index=False)
        end = time.time()
        elapsed_time = round(end-start, 2)
        print('Elapsed time:', elapsed_time)
    return

def buildNewSeasonAvg(source: pd.DataFrame, df: pd.DataFrame, _dir):
    print('Creating new seasonAvg...')
    all_avgs = []
    for index, row in source.iterrows():
        pid = row['p_id']
        wy = row['wy']
        year = int(wy.split(" | ")[1])
        if wy in df['wy'].values:
            start = df.loc[df['wy']==wy].index.values[0]
            stats = df.loc[
                (df['p_id']==pid)&
                (df['wy'].str.contains(str(year)))&
                (df.index<start),
                'points'
            ].values
        else:
            stats = df.loc[
                (df['p_id']==pid)&
                (df['wy'].str.contains(str(year))),
                'points'
            ].values
        if len(stats) > 1:
            s_avg = np.mean(stats)
        else:
            s_avg = np.NaN
        all_avgs.append(s_avg)
    source['seasonAvg'] = all_avgs
    source.fillna(0, inplace=True)
    return source

##########################

# df = pd.read_csv("%s.csv" % "../../../../data/fantasyData")

# source = pd.read_csv("%s.csv" % "../source/new_source")

# # buildSeasonAvg(source, df, './')

# buildNewSeasonAvg(source, df, './')