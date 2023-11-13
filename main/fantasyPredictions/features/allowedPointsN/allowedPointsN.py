import pandas as pd
import numpy as np
import os
import multiprocessing
import time
from functools import partial
import regex as re

def func(source: pd.DataFrame, n):
    ROOT_DIR = os.path.abspath(__file__)
    n_idx = ROOT_DIR.index('NFLPredictions3')
    ROOT_DIR = ROOT_DIR[:n_idx-1]
    df = pd.read_csv("%s.csv" % (ROOT_DIR + "/NFLPredictions3/data/fantasyData"))
    cd = pd.read_csv("%s.csv" % (ROOT_DIR + "/NFLPredictions3/data/gameData"))
    cols = [('allowedPointsN_' + str(i)) for i in range(n)]
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    for index, row in source.iterrows():
        position = row['position']
        wy = row['wy']
        abbr = row['abbr']
        key = row['key']
        all_abbrs = cd.loc[cd['key']==key, ['home_abbr', 'away_abbr']].values[0]
        opp_abbr = list(set(all_abbrs).difference(set([abbr])))[0]
        start = cd.loc[cd['wy']==wy].index.values[0]
        keys = cd.loc[(cd.index<start)&((cd['home_abbr']==opp_abbr)|(cd['away_abbr']==opp_abbr)), 'key'].values[n*-1:]
        stats = df.loc[
            (df['key'].isin(keys))&
            (df['abbr']!=opp_abbr)&
            (df['position']==position)
        ]
        stats = stats.groupby('key')['points'].mean().values
        stats = np.flip(stats)
        if len(stats) == 0:
            stats = np.zeros(n)
        if len(stats) < n:
            dif = n - len(stats)
            stats = np.concatenate((stats, np.zeros(dif)))
        new_df.loc[len(new_df.index)] = list(row.values) + list(stats)
    return new_df

def buildAllowedPointsN(n, source: pd.DataFrame, _dir):
    fn = 'allowedPointsN_' + str(n)
    if (fn + '.csv') in os.listdir(_dir):
        print(fn + ' already created.')
        return
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    source_split = np.array_split(source, num_partitions)
    df_list = []
    if __name__ == 'fantasyPredictions.features.allowedPointsN.allowedPointsN' or __name__ == '__main__':
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

def buildNewAllowedPointsN(n, source: pd.DataFrame, df: pd.DataFrame, cd: pd.DataFrame, _dir):
    print('Creating new allowedPointsN_' + str(n) + '...')
    cols = [('allowedPointsN_' + str(i)) for i in range(n)]
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    for index, row in source.iterrows():
        position = row['position']
        wy = row['wy']
        abbr = row['abbr']
        key = row['key']
        all_abbrs = list(set(source.loc[source['key']==key, 'abbr'].values))
        opp_abbr = list(set(all_abbrs).difference(set([abbr])))[0]
        if wy in cd['wy'].values:
            start = cd.loc[cd['wy']==wy].index.values[0]
            keys = cd.loc[(cd.index<start)&((cd['home_abbr']==opp_abbr)|(cd['away_abbr']==opp_abbr)), 'key'].values[n*-1:]
        else:
            keys = cd.loc[((cd['home_abbr']==opp_abbr)|(cd['away_abbr']==opp_abbr)), 'key'].values[n*-1:]
        stats = df.loc[
            (df['key'].isin(keys))&
            (df['abbr']!=opp_abbr)&
            (df['position']==position)
        ]
        stats = stats.groupby('key')['points'].mean().values
        stats = np.flip(stats)
        if len(stats) == 0:
            stats = np.zeros(n)
        if len(stats) < n:
            dif = n - len(stats)
            stats = np.concatenate((stats, np.zeros(dif)))
        new_df.loc[len(new_df.index)] = list(row.values) + list(stats)
    return new_df

#######################

# source = pd.read_csv("%s.csv" % "../source/new_source")
# df = pd.read_csv("%s.csv" % "../../../../data/fantasyData")
# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")

# buildNewAllowedPointsN(5, source, df, cd, './')
