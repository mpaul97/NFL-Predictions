from distutils.command.build import build
import pandas as pd
import numpy as np
import os

DATA_PATH = "../../../rawData/"

def buildTarget():
    
    df = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    
    start = df.loc[df['wy'].str.contains('1980')].index[0]
    
    df = df.loc[df.index>=start]
    
    keys, opps, wys, won, points = [], [], [], [], [];
    
    for index, row in df.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_key = home_abbr + "-" + key
        away_key = away_abbr + "-" + key
        keys.append(home_key)
        opps.append(away_abbr)
        keys.append(away_key)
        opps.append(home_abbr)
        wys.append(row['wy'])
        wys.append(row['wy'])
        if home_abbr == row['winning_abbr']:
            won.append(1)
            won.append(0)
        else:
            won.append(0)
            won.append(1)
        points.append(row['home_points'])
        points.append(row['away_points'])
        
    new_df = pd.DataFrame()
    new_df['key'] = keys
    new_df['opp_abbr'] = opps
    new_df['wy'] = wys
    new_df['won'] = won
    new_df['points'] = points
    
    new_df.to_csv("%s.csv" % "target", index=False)
    
####################################

buildTarget()