import pandas as pd
import numpy as np
import os

def getPoints(n, abbr, wy, cd: pd.DataFrame):
    try:
        start = cd.loc[cd['wy']==wy].index.values[0]
        stats = cd.loc[
            (cd.index<start)&
            ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr)),
            ['home_abbr', 'home_points', 'away_points']
        ].values[-n:]
    except IndexError:
        stats = cd.loc[
            ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr)),
            ['home_abbr', 'home_points', 'away_points']
        ].values[-n:]
    points = [row[1] if (row[0] == abbr) else row[-1] for row in stats]
    points = points[::-1]
    if len(points) < n:
        dif = n - len(points)
        points += [np.nan for _ in range(dif)]
    return points

def buildPointsN(n, source: pd.DataFrame, cd: pd.DataFrame, _dir):
    if 'pointsN_' + str(n) + '.csv' in os.listdir(_dir):
        print('pointsN_' + str(n) + '.csv already created.')
        return
    print('Creating pointsN_' + str(n) + '...')
    cols = [(pre + 'pointsN_' + str(i)) for pre in ['home_', 'away_'] for i in range(n)]
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    for index, row in source.iterrows():
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_points = getPoints(n, home_abbr, wy, cd)
        away_points = getPoints(n, away_abbr, wy, cd)
        new_df.loc[len(new_df.index)] = list(row.values) + home_points + away_points
    new_df.fillna(new_df.mean(), inplace=True)
    new_df.to_csv("%s.csv" % (_dir + "pointsN_" + str(n)), index=False)
    return

def buildNewPointsN(n, source: pd.DataFrame, cd: pd.DataFrame, _dir):
    print('Creating new pointsN_' + str(n) + '...')
    cols = [(pre + 'pointsN_' + str(i)) for pre in ['home_', 'away_'] for i in range(n)]
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    for index, row in source.iterrows():
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_points = getPoints(n, home_abbr, wy, cd)
        away_points = getPoints(n, away_abbr, wy, cd)
        new_df.loc[len(new_df.index)] = list(row.values) + home_points + away_points
    new_df.fillna(new_df.mean(), inplace=True)
    
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + "newPointsN_" + str(n)), index=False)
    
    return new_df

#############################

# source = pd.read_csv("%s.csv" % "../source/source")
# cd = pd.read_csv("%s.csv" % "../../../../data/oldGameData_78")

# buildPointsN(10, source, cd, "./")