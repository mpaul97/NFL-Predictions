import pandas as pd
import numpy as np
import os

DATA_PATH = "../../../../rawData/"
PLAYER_PATH = DATA_PATH + "positionData/"
SOURCE_PATH = "../features/joining/"

def build():
    
    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    sd = pd.read_csv("%s.csv" % (DATA_PATH + "starterIDs_78-21"))
    df = pd.read_csv("%s.csv" % (PLAYER_PATH + "QBData"))
    
    dfList = []
    
    for index, row in source.iterrows():
        abbr =  row['key'].split("-")[0]
        wy = row['wy']
        # print(abbr, wy)
        players = sd.loc[(sd['abbr']==abbr)&(sd['wy']==wy), 'players'].values[0].split(" | ")
        temp = df.loc[(df['abbr']==abbr)&(df['wy']==wy)]
        if len(temp.index) > 1:
            tempStats = []    
            for p in players:
                temp1 = temp.loc[temp['p_id']==p]
                if not temp1.empty:
                    tempStats.append(temp1)
            temp_df = pd.concat(tempStats)
            temp_df = temp_df.sort_values(by="attempted_passes", ascending=False)
            dfList.append(temp_df.head(1))
        else:
            if not temp.empty:
                dfList.append(temp)
            else:
                print("->", abbr, wy)
                dfList.append(pd.DataFrame({'p_id': ['UNK']}))
                
    new_df = pd.concat(dfList)
    new_df.insert(0, 'key', list(source['key']))
    new_df.insert(1, 'opp_abbr', list(source['opp_abbr']))
    new_df.fillna(new_df.mean()/1.5, inplace=True)
    
    new_df.to_csv("%s.csv" % "target", index=False)
    
###########################

build()