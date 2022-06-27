import pandas as pd
import numpy as np
import multiprocessing
import os

DATA_PATH = "../rawData/"
SCRAPE_PATH = "scrapeStarters/"

def func(df):
    
    keys, merged = [], []
    
    for index, row in df.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_key = home_abbr + "-" + key
        away_key = away_abbr + "-" + key
        keys.append(home_key)
        keys.append(away_key)
        home_starters = pd.read_csv("%s.csv" % (SCRAPE_PATH + home_key))
        away_starters = pd.read_csv("%s.csv" % (SCRAPE_PATH + away_key))
        temp_home = []
        for index1, row1 in home_starters.iterrows():
            starter = row1['starters']
            position = row1['positions']
            temp_home.append(str(starter) + ":" + str(position))
        merged.append('|'.join(temp_home))
        temp_away = []
        for index1, row1 in away_starters.iterrows():
            starter = row1['starters']
            position = row1['positions']
            temp_away.append(str(starter) + ":" + str(position))
        merged.append('|'.join(temp_away))
    
    new_df = pd.DataFrame()
    new_df['key'] = keys
    new_df['merged'] = merged
    
    return new_df

def mergeScrape():
    
    df = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    
    df_list = []
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    df_split = np.array_split(df, num_partitions)
    
    if __name__ == '__main__':
        pool = multiprocessing.Pool(num_cores)
        all_dfs = pd.concat(pool.map(func, df_split))
        df_list.append(all_dfs)
        pool.close()
        pool.join()
        
    if df_list:
        new_df = pd.concat(df_list)
        new_df.to_csv("%s.csv" % ('mergedScrapeStarters'), index=False)
    
########################

mergeScrape()