import pandas as pd
import numpy as np
import os
from random import randrange

PBP_PATH = "../playByPlay/pbpTables/"

def collect():
    
    years = [i for i in range(1994, 2022)]
    
    df_list = []
    
    for year in years:
        fns = [fn for fn in os.listdir(PBP_PATH) if str(year) in fn]
        rand_index = randrange(0, len(fns))
        df = pd.read_csv(PBP_PATH + fns[rand_index], compression='gzip')
        df.insert(0, 'key', fns[rand_index].replace(".csv.gz",""))
        df_list.append(df)
        
    new_df = pd.concat(df_list)
    
    new_df.to_csv("%s.csv" % "pbpTestData", index=False)
    
    return

def filterLines():
    
    df = pd.read_csv("%s.csv" % "pbpTestData")
    
    lines = df['Detail'].values
    
    lengths = []
    
    for line in lines:
        lengths.append(len(line))
        
    lengths = list(set(lengths))
    
    lengths.sort()
    
    # lengths = lengths[::2]
    
    new_df = pd.DataFrame(columns=['line'])
    
    for length in lengths:
        line = [line for line in lines if len(line) == length][0]
        new_df.loc[len(new_df.index)] = [line]
        
    new_df.to_csv("%s.csv" % "testLines", index=False)
    
    return

##############################

# collect()

filterLines()