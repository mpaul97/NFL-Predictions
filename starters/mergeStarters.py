import pandas as pd
import numpy as np
import os

pn = pd.read_csv("%s.csv" % "../playerNames/finalPlayerInfo")

def updatePositions(df: pd.DataFrame):
    sp = pd.read_csv("%s.csv" % "scrapeSimplePositions")
    sp.drop(columns=['frequency'], inplace=True)
    df.columns = ['p_id', 'position']
    temp_df = df.merge(sp, on=['position'])
    df['position'] = temp_df['simplePosition']
    return df

def mergeStarters():
    cd = pd.read_csv("%s.csv" % "../data/gameData")
    _dir = "scrapeStarters/"
    cols = ['key', 'abbr', 'starters']
    new_df = pd.DataFrame(columns=cols)
    all_unks = [] # unk pids
    for index, row in cd.iterrows():
        print(row['wy'])
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_fn = home_abbr + "-" + key
        away_fn = away_abbr + "-" + key
        home_df = pd.read_csv("%s.csv" % (_dir + home_fn))
        home_df = updatePositions(home_df)
        away_df = pd.read_csv("%s.csv" % (_dir + away_fn))
        away_df = updatePositions(away_df)
        home_starters = '|'.join([(row1['p_id'] + ':' + row1['position']) for _, row1 in home_df.iterrows()])
        away_starters = '|'.join([(row1['p_id'] + ':' + row1['position']) for _, row1 in away_df.iterrows()])
        new_df.loc[len(new_df.index)] = [key, home_abbr, home_starters]
        new_df.loc[len(new_df.index)] = [key, away_abbr, away_starters]
        # --------------------------------
        # get unk pids
        unk_df = pd.concat([home_df, away_df])
        unk_df = unk_df.loc[unk_df['position']=='UNK']
        if not unk_df.empty:
            all_unks.append(unk_df)
    new_df.to_csv("%s.csv" % "allStarters", index=False)
    pd.concat(all_unks).to_csv("%s.csv" % "unkStarters", index=False)
    return

def replaceUnknowns():
    all_df = pd.read_csv("%s.csv" % "allStarters")
    df = pd.read_csv("%s.csv" % "unkStarters")
    df.drop_duplicates(inplace=True)
    for index, row in df.iterrows():
        pid = row['p_id']
        position = pn.loc[pn['p_id']==pid, 'position'].values[0]
        old_starter = pid + ':UNK'
        temp_df = all_df.loc[all_df['starters'].str.contains(old_starter)]
        for index1, row1 in temp_df.iterrows():
            starters = row1['starters']
            starters = starters.replace(old_starter, (pid + ":" + position))
            all_df.at[index1, 'starters'] = starters
    all_df.to_csv("%s.csv" % "allStarters", index=False)
    return

def addWys():
    df = pd.read_csv("%s.csv" % "allStarters")
    cd = pd.read_csv("%s.csv" % "../data/gameData")
    wys = []
    for index, row in df.iterrows():
        key = row['key']
        wy = cd.loc[cd['key']==key, 'wy'].values[0]
        print(wy)
        wys.append(wy)
    df.insert(2, 'wy', wys)
    df.to_csv("%s.csv" % "allStarters", index=False)
    return

#########################

# mergeStarters()

# replaceUnknowns()

addWys()