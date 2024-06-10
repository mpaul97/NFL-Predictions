import pandas as pd
import numpy as np
import os
import multiprocessing
import regex as re
from functools import partial
import time

def func(source: pd.DataFrame):
    ROOT_DIR = os.path.abspath(__file__)
    n_idx = ROOT_DIR.index('NFLPredictions3')
    ROOT_DIR = ROOT_DIR[:n_idx-1]
    df = pd.read_csv("%s.csv" % (ROOT_DIR + "/NFLPredictions3/data/fantasyData"))
    all_avgs = []
    for index, row in source.iterrows():
        pid = row['p_id']
        wy = row['wy']
        start = df.loc[df['wy']==wy].index.values[0]
        stats = df.loc[
            (df['p_id']==pid)&
            (df.index<start),
            'points'
        ].values
        if len(stats) > 1:
            s_avg = np.mean(stats)
        else:
            s_avg = np.NaN
        all_avgs.append(s_avg)
    source['careerAvg'] = all_avgs
    return source

def buildCareerAvg(source: pd.DataFrame, _dir):
    if 'careerAvg.csv' in os.listdir(_dir):
        print('careerAvg already created.')
        return
    print('Creating careerAvg...')
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    source_split = np.array_split(source, num_partitions)
    df_list = []
    if __name__ == 'fantasyPredictions.features.careerAvg.careerAvg':
        start = time.time()
        pool = multiprocessing.Pool(num_cores)
        all_dfs = pd.concat(pool.map(func, source_split))
        df_list.append(all_dfs)
        pool.close()
        pool.join()
        new_df = pd.concat(df_list)
        new_df.fillna(0, inplace=True)
        new_df.to_csv("%s.csv" % (_dir + "careerAvg"), index=False)
        end = time.time()
        elapsed_time = round(end-start, 2)
        print('Elapsed time:', elapsed_time)
    return

def buildNewCareerAvg(source: pd.DataFrame, df: pd.DataFrame, _dir):
    print('Creating new careerAvg...')
    all_avgs = []
    for index, row in source.iterrows():
        pid = row['p_id']
        wy = row['wy']
        year = int(wy.split(" | ")[1])
        if wy in df['wy'].values:
            start = df.loc[df['wy']==wy].index.values[0]
            stats = df.loc[
                (df['p_id']==pid)&
                (df.index<start),
                'points'
            ].values
        else:
            stats = df.loc[
                (df['p_id']==pid),
                'points'
            ].values
        if len(stats) > 1:
            s_avg = np.mean(stats)
        else:
            s_avg = np.NaN
        all_avgs.append(s_avg)
    source['careerAvg'] = all_avgs
    source.fillna(0, inplace=True)
    return source

###########################