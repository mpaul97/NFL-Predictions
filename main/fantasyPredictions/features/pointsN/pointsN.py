import pandas as pd
import numpy as np
import os
import multiprocessing
import time
from functools import partial

def func(source: pd.DataFrame, n):
    ROOT_DIR = os.path.abspath(__file__)
    n_idx = ROOT_DIR.index('NFLPredictions3')
    ROOT_DIR = ROOT_DIR[:n_idx-1]
    df = pd.read_csv("%s.csv" % (ROOT_DIR + "/NFLPredictions3/data/fantasyData"))
    cols = [('pointsN_' + str(i)) for i in range(n)]
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    for index, row in source.iterrows():
        pid = row['p_id']
        wy = row['wy']
        start = df.loc[(df['p_id']==pid)&(df['wy']==wy)].index.values[0]
        stats = df.loc[(df['p_id']==pid)&(df.index<start), 'points'].values[n*-1:]
        stats = np.flip(stats)
        if len(stats) == 0:
            stats = np.zeros(n)
        if len(stats) < n:
            dif = n - len(stats)
            stats = np.concatenate((stats, np.zeros(dif)))
        new_df.loc[len(new_df.index)] = list(row.values) + list(stats)
    return new_df

def buildPointsN(n, source: pd.DataFrame, _dir):
    fn = 'pointsN_' + str(n)
    if (fn + '.csv') in os.listdir(_dir):
        print(fn + ' already created.')
        return
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    source_split = np.array_split(source, num_partitions)
    df_list = []
    if __name__ == 'fantasyPredictions.features.pointsN.pointsN' or __name__ == '__main__':
        print('Creating ' + fn + '...')
        start = time.time()
        pool = multiprocessing.Pool(num_cores)
        all_dfs = pd.concat(pool.map(partial(func, n=n), source_split))
        df_list.append(all_dfs)
        pool.close()
        pool.join()
        new_df = pd.concat(df_list)
        new_df.fillna(0, inplace=True)
        new_df.to_csv("%s.csv" % (_dir + fn), index=False)
        end = time.time()
        elapsed_time = round(end-start, 2)
        print('Elapsed time:', elapsed_time)
    return

def buildNewPointsN(n, source: pd.DataFrame, df: pd.DataFrame, _dir):
    print('Creating new pointsN_' + str(n) + '...')
    cols = [('pointsN_' + str(i)) for i in range(n)]
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    for index, row in source.iterrows():
        pid = row['p_id']
        wy = row['wy']
        if wy in df['wy'].values:
            start = df.loc[(df['p_id']==pid)&(df['wy']==wy)].index.values[0]
            stats = df.loc[(df['p_id']==pid)&(df.index<start), 'points'].values[n*-1:]
        else:
            stats = df.loc[df['p_id']==pid, 'points'].values[n*-1:]
        stats = np.flip(stats)
        if len(stats) == 0:
            stats = np.zeros(n)
        if len(stats) < n:
            dif = n - len(stats)
            stats = np.concatenate((stats, np.zeros(dif)))
        new_df.loc[len(new_df.index)] = list(row.values) + list(stats)
    return new_df

##########################

# df = pd.read_csv("%s.csv" % "../../../../data/fantasyData")
# source = pd.read_csv("%s.csv" % "../source/new_source")

# df = buildNewPointsN(5, source, df, './')

# df.to_csv("%s.csv" % "newPoints_5", index=False)

# buildPointsN(5, source, './')