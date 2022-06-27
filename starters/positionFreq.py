import pandas as pd
import numpy as np
import os

DATA_PATH = "../rawData/"
SCRAPE_PATH = "scrapeStarters/"

def getPositionFreq():

    cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))

    pos_keys = []

    for index, row in cd.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_key = home_abbr + "-" + key
        away_key = away_abbr + "-" + key
        home_starters = pd.read_csv("%s.csv" % (SCRAPE_PATH + home_key))
        away_starters = pd.read_csv("%s.csv" % (SCRAPE_PATH + away_key))
        home_poses = home_starters['positions'].values
        away_poses = away_starters['positions'].values
        for i, pos in enumerate(home_poses):
            pos_keys.append(str(pos) + ":" + str(i))
        for i, pos in enumerate(away_poses):
            pos_keys.append(str(pos) + ":" + str(i))
        
    unique_keys = list(set(pos_keys))
    
    counts = []
    
    for key in unique_keys:
        counts.append((key, pos_keys.count(key)))
        
    counts.sort(key=lambda x: x[1], reverse=True)
        
    df = pd.DataFrame()
    df['key'] = [c[0] for c in counts]
    df['num'] = [c[1] for c in counts]
    df['total'] = [len(cd.index)*2 for i in range(len(counts))]
    
    df.to_csv("%s.csv" % ('positionFrequencies'), index=False)

def cleanPositionFreq():
    
    df = pd.read_csv("%s.csv" % ('positionFrequencies'))
    
    indexes = [key.split(":")[1] for key in list(df['key'])]
    df.insert(0, 'k_index', indexes)
    
    df.sort_values(by=['k_index'], inplace=True)
    
    df.to_csv("%s.csv" % ('positionFrequencies'), index=False)

###########################

# getPositionFreq()

cleanPositionFreq()