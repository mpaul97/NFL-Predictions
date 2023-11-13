import pandas as pd
import numpy as np
import os
from ordered_set import OrderedSet

def calc(df):
    
    pointsList = []
    
    for index, row in df.iterrows():
        print(row['wy'])
        points = 0
        for col in df.columns:
            if 'passing_touchdowns' in col:
                ptds = round(row[col], 0)
                points += ptds*4
            if 'passing_yards' in col:
                pyds = round(row[col], 0)
                points += pyds*0.04
                if pyds > 300:
                    points += 3
            if 'interceptions' in col:
                ints = round(row[col], 0)
                points -= ints
            if 'rush_yards' in col:
                ryds = round(row[col], 0)
                points += ryds*0.1
                if ryds > 100:
                    points += 3
            if 'rush_touchdowns' in col:
                rtds = round(row[col], 0)
                points += rtds*6
            if 'receptions' in col:
                recs = round(row[col], 0)
                points += recs
            if 'receiving_yards' in col:
                recyds = round(row[col], 0)
                points += recyds*0.1
                if recyds > 100:
                    points += 3
            if 'receiving_touchdowns' in col:
                rectds = round(row[col], 0)
                points += rectds*6
        pointsList.append(round(points, 2))
        
    return pointsList

def buildKings():
    
    _dir = "positionData/"

    files = os.listdir(_dir)
    
    keep_cols = ['p_id', 'abbr', 'game_key', 'wy', 'position']

    for fn in files:
        if ".csv" in fn and ('QB' in fn or 'RB' in fn or 'WR' in fn or 'TE' in fn) and 'Returns' not in fn:
            df = pd.read_csv(_dir + fn)
            drop_cols = set(df.columns).difference(set(keep_cols))
            points = calc(df)
            df.drop(columns=drop_cols, inplace=True)
            df['points'] = points
            df.to_csv(("positionKings/" + fn.replace("Data","Kings")), index=False)

def addWeekRanks():
    
    _dir = "positionKings/"
    
    files = os.listdir(_dir)
    
    files = [f for f in files if 'csv' in f and 'skill' not in f]
    
    for fn in files:
        df = pd.read_csv(_dir + fn)
        wys = list(OrderedSet(df['wy'].values))
        df_list = []
        for wy in wys:
            print(fn, wy)
            stats = df.loc[df['wy']==wy].sort_values(by=['points'], ascending=False).reset_index(drop=True)
            stats['week_rank'] = stats.index + 1
            df_list.append(stats)
        new_df = pd.concat(df_list)
        new_df = df.merge(new_df, on=['p_id', 'abbr', 'game_key', 'wy', 'position', 'points'], how='left')
        new_df.to_csv((_dir + fn), index=False)
        
    return

##########################

# buildKings()

# addWeekRanks()