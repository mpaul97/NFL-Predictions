import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import os

from paths import DATA_PATH, POSITION_PATH

def buildTeamPidsByWeek():
    
    df = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))
    
    cd_list = []
    for fn in os.listdir(POSITION_PATH):
        if 'csv' in fn:
            cd_list.append(pd.read_csv(POSITION_PATH + fn))
    cd = pd.concat(cd_list)
    
    new_df = pd.DataFrame(columns=['key', 'wy', 'abbr', 'pids'])
    
    for index, row in df.iterrows():
        key = row['key']
        wy = row['wy']
        print(wy)
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_pids = cd.loc[(cd['game_key']==key)&(cd['abbr']==home_abbr), 'p_id'].values
        away_pids = cd.loc[(cd['game_key']==key)&(cd['abbr']==away_abbr), 'p_id'].values
        new_df.loc[len(new_df.index)] = [key, wy, home_abbr, '|'.join(home_pids)]
        new_df.loc[len(new_df.index)] = [key, wy, away_abbr, '|'.join(away_pids)]
        
    new_df.to_csv("%s.csv" % "teamPidsByWeek", index=False)
    
    return

####################

buildTeamPidsByWeek()