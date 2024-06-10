import pandas as pd
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
import statsmodels.api as sm
from random import randrange
import regex as re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

pd.options.mode.chained_assignment = None

class Grade:
    def __init__(self, pid, grade):
        self.pid = pid
        self.grade = grade
        self.all_grades = []
        self.all_grades.append(grade)
    def show(self):
        print(self.__dict__)
        return
    def getEndGrade(self):
        # return (self.grade * 0.75) + (0.25 * 101)
        return 100
    
# END CLASSES

pg_train_cols = {
    'qb': [
        'passing_yards', 'passing_touchdowns', 'interceptions_thrown',
        'quarterback_rating', 'rush_yards', 'rush_touchdowns',
        'completion_percentage', 'adjusted_yards_per_attempt', 'times_sacked',
        'attempted_passes'
    ]
}

position_sizes = {
    'QB': 1, 'RB': 2, 'WR': 4,
    'TE': 1
}

# zero division
def zeroDivision(num, dem):
    try:
        return num/dem
    except ZeroDivisionError:
        return 0

# flatten list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]

# build per game train data for every week
def buildAllPerGamePlayerTrainData(position, df: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    train_dir = _dir + 'perGameTrain/'
    
    # df = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))
    # cd = pd.read_csv("%s.csv" % (POSITION_PATH + position + "Data"))
    
    wys = list(set(df['wy'].values))
        
    cd: pd.DataFrame = cd.loc[cd['wy'].isin(wys)]
    cd.reset_index(drop=True, inplace=True)
    cd: pd.DataFrame = cd[['p_id', 'wy', 'game_key', 'abbr']+pg_train_cols[position.lower()]]
    
    # new_cols = ['pointsFor', 'pointsAgainst', 'won', 'total_pass_attempts']
    pointsFor, pointsAgainst, wins, passAttempts, passAttemptsPerc = [], [], [], [], []
    
    for index, row in cd.iterrows():
        key = row['game_key']
        print(index, len(cd.index))
        abbr = row['abbr']
        game = df.loc[df['key']==key]
        home_abbr = game['home_abbr'].values[0]
        away_abbr = game['away_abbr'].values[0]
        winning_abbr = game['winning_abbr'].values[0]
        home_points = game['home_points'].values[0]
        away_points = game['away_points'].values[0]
        home_pass_attempts = game['home_pass_attempts'].values[0]
        away_pass_attempts = game['away_pass_attempts'].values[0]
        wins.append(1) if winning_abbr == abbr else wins.append(0)
        attempted_passes = row['attempted_passes']
        if abbr == home_abbr:
            pointsFor.append(home_points)
            pointsAgainst.append(away_points)
            passAttempts.append(home_pass_attempts)
            if attempted_passes == 0:
                passAttemptsPerc.append(0)
            else:
                passAttemptsPerc.append(zeroDivision(attempted_passes, home_pass_attempts))
        else:
            pointsFor.append(away_points)
            pointsAgainst.append(home_points)
            passAttempts.append(away_pass_attempts)
            if attempted_passes == 0:
                passAttemptsPerc.append(0)
            else:
                passAttemptsPerc.append(zeroDivision(attempted_passes, away_pass_attempts))
            
    cd['pointsFor'] = pointsFor
    cd['pointsAgainst'] = pointsAgainst
    cd['won'] = wins
    cd['totalPassAttempts'] = passAttempts
    cd['passAttemptsPercent'] = passAttemptsPerc
    
    cd = cd.round(2)
    
    cd.to_csv("%s.csv" % (_dir + train_dir + position + "_train_all"), index=False)
    
    return

# predict
def predictPerGameGrades(position, wys, _dir):
    
    print('Predicting perGame grades...')
    
    train_dir = _dir + 'perGameTrain/'
    target_dir = _dir + 'perGameTargets/'
    
    train_list, target_list = [], []
    for wy in wys:
        fn_wy = wy.replace(' | ','-')
        train_list.append(pd.read_csv("%s.csv" % (train_dir + position + '_train_' + fn_wy)))
        target_list.append(pd.read_csv("%s.csv" % (target_dir + position + '_target_' + fn_wy)))
    
    train = pd.concat(train_list)
    target = pd.concat(target_list)
    
    test = pd.read_csv("%s.csv" % (train_dir + position + '_train_all'))
    
    data = train.merge(target, on=['p_id'])
    
    X = data.drop(columns=['p_id', 'grade'])
    y = data['grade']
    
    all_models = []
    for i in range(50):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=randrange(1, 42))
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)

        all_models.append((acc, model))
        
    all_models.sort(key=lambda x: x[0], reverse=True)
    
    best_model: LinearRegression = all_models[0][1]
    print('Per Game accuracy:', all_models[0][0])
    
    z_test = test.drop(columns=['p_id', 'wy', 'game_key', 'abbr'])
    
    preds = best_model.predict(z_test)
    
    test['grade'] = preds
    test = test[['p_id', 'wy', 'game_key', 'abbr', 'grade']]
    test = test.round(4)
    
    test.to_csv("%s.csv" % (_dir + position + "_perGame_grades"), index=False)
    
    return

# create grades for all pids of position, sum of per game grades per season
def createFinalGrades(position, df: pd.DataFrame, cd: pd.DataFrame, gdf: pd.DataFrame, _dir):
    
    print("Creating " + position + "_finalGrades...")
    
    cd = cd.loc[cd['game_key'].isin(df['key'].values)]
    all_pids = list(set(cd['p_id'].values))
    
    new_df = pd.DataFrame(columns=['p_id', 'wy', 'grade'])
    
    mean = np.mean(gdf['grade'].values)
    
    for pid in all_pids:
        g = Grade(pid, 100)
        data: pd.DataFrame = gdf.loc[gdf['p_id']==pid]
        data.reset_index(drop=True, inplace=True)
        for index, row in data.iterrows():
            wy = row['wy']
            year = int(wy.split(" | ")[1])
            if index != len(data.index) - 1:
                next_wy = data.iloc[index+1]['wy']
                next_year = int(next_wy.split(" | ")[1])
                if year != next_year: # new season
                    # normal grade for curr week
                    grade = row['grade']
                    grade = (grade - 3) if grade < mean else grade
                    grade = (grade - mean)*5
                    g.grade += grade
                    new_df.loc[len(new_df.index)] = [pid, wy, g.grade]
                    # new year grade
                    g.grade = g.getEndGrade()
                    new_df.loc[len(new_df.index)] = [pid, str(year), g.grade]
                else:
                    grade = row['grade']
                    grade = (grade - 3) if grade < mean else grade
                    grade = (grade - mean)*5
                    g.grade += grade
                    new_df.loc[len(new_df.index)] = [pid, wy, g.grade]
            else:
                grade = row['grade']
                grade = (grade - 3) if grade < mean else grade
                grade = (grade - mean)*5
                g.grade += grade
                new_df.loc[len(new_df.index)] = [pid, wy, g.grade]
                if index == len(data.index) - 1:
                    g.grade = g.getEndGrade()
                    new_df.loc[len(new_df.index)] = [pid, str(year), g.grade]
    
    new_df.to_csv("%s.csv" % (_dir + position + "_finalGrades"), index=False)
                    
    return new_df

# get past week grades for given pids
def getGrade(pid, wy, data: pd.DataFrame):
    try:
        temp_df: pd.DataFrame = data.loc[data['p_id']==pid]
        temp_df.reset_index(drop=True, inplace=True)
        idx = temp_df.loc[temp_df['wy']==wy].index.values[0]
        if idx == 0:
            grade = 100
        else:
            grade = temp_df.iloc[idx-1]['grade']
    except IndexError as error:
        print(pid, wy, error)
        return 100
    return grade

# build all grades
def buildPlayerGrades(position, source: pd.DataFrame, df: pd.DataFrame, sdf: pd.DataFrame, POSITION_PATH, buildNew, _dir):
    
    # PREPROCESSING
    train_dir = _dir + 'perGameTrain/'
    train_wys = [(re.findall(r"[0-9]-[0-9]{4}", fn)[0]).replace('-', ' | ') for fn in os.listdir(train_dir) if 'all' not in fn]
    
    cd = pd.read_csv("%s.csv" % (POSITION_PATH + position + 'Data'))
    
    # buildAllPerGamePlayerTrainData(position, df, cd, _dir)
    
    if buildNew:
        print('Creating new ' + position + '_perGame_grades & ' + position + '_finalGrades...')
        predictPerGameGrades(position, train_wys, _dir)
        gdf = pd.read_csv("%s.csv" % (_dir + position + "_perGame_grades"))
        data: pd.DataFrame = createFinalGrades(position, df, cd, gdf, _dir)
    else:
        print('Using existing ' + position + '_playerGrades')
        return pd.read_csv("%s.csv" % (_dir + position + "_playerGrades"))
    # END PREPROCESSING
    
    print('Creating ' + position + '_playerGrades...')
    
    position_size = position_sizes[position]
    cols = [[('home_grade_' + position + '_' + str(i)), ('away_grade_' + position + '_' + str(i))] for i in range(position_size)]
    cols = flatten(cols)
    
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_starters = (sdf.loc[(sdf['abbr']==home_abbr)&(sdf['wy']==wy), 'starters'].values[0]).split("|")
        away_starters = (sdf.loc[(sdf['abbr']==away_abbr)&(sdf['wy']==wy), 'starters'].values[0]).split("|")
        home_pids = [s.replace((':'+position),'') for s in home_starters if position in s][:position_size]
        away_pids = [s.replace((':'+position),'') for s in away_starters if position in s][:position_size]
        home_stats, away_stats = [], []
        for pid in home_pids:
            grade = getGrade(pid, wy, data)
            home_stats.append(grade)
        for pid in away_pids:
            grade = getGrade(pid, wy, data)
            away_stats.append(grade)
        home_stats = [100 for _ in range(position_size)] if len(home_pids) == 0 else home_stats
        away_stats = [100 for _ in range(position_size)] if len(away_pids) == 0 else away_stats
        new_df.loc[len(new_df.index)] = list(row.values) + home_stats + away_stats
        
    new_df.to_csv("%s.csv" % (_dir + position + "_playerGrades"), index=False)
    
    return

# build new player grades
def buildNewPlayerGrades(position, source: pd.DataFrame, sdf: pd.DataFrame, _dir):
    
    data = pd.read_csv("%s.csv" % (_dir + position + "_finalGrades"))
    
    position_size = position_sizes[position]
    cols = [[('home_grade_' + position + '_' + str(i)), ('away_grade_' + position + '_' + str(i))] for i in range(position_size)]
    cols = flatten(cols)
    
    new_df = pd.DataFrame(columns=list(source.columns)+cols)
    
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_starters = (sdf.loc[(sdf['abbr']==home_abbr)&(sdf['wy']==wy), 'starters'].values[0]).split("|")
        away_starters = (sdf.loc[(sdf['abbr']==away_abbr)&(sdf['wy']==wy), 'starters'].values[0]).split("|")
        home_pids = [s.replace((':'+position),'') for s in home_starters if position in s][:position_size]
        away_pids = [s.replace((':'+position),'') for s in away_starters if position in s][:position_size]
        home_stats, away_stats = [], []
        for pid in home_pids:
            if wy in data['wy'].values:
                grade = getGrade(pid, wy, data)
            else:
                if week != 1:
                    # get last grade that is not 100 (init val)
                    past_grades = data.loc[data['p_id']==pid, 'grade'].values[-2:]
                    if len(past_grades) != 0:
                        grade = [g for g in past_grades if g != 100][0]
                    else:
                        grade = 100
                else:
                    grade = 100
            home_stats.append(grade)
        for pid in away_pids:
            if wy in data['wy'].values:
                grade = getGrade(pid, wy, data)
            else:
                if week != 1:
                    # get last grade that is not 100 (init val)
                    past_grades = data.loc[data['p_id']==pid, 'grade'].values[-2:]
                    if len(past_grades) != 0:
                        grade = [g for g in past_grades if g != 100][0]
                    else:
                        grade = 100
                else:
                    grade = 100
            away_stats.append(grade)
        new_df.loc[len(new_df.index)] = list(row.values) + home_stats + away_stats
        
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + position + "_newPlayerGrades"), index=False)
        
    return new_df

########################################

# position = 'QB'
# source = pd.read_csv("%s.csv" % "../source/source")
# df = pd.read_csv("%s.csv" % "../../../../data/gameData")
# sdf = pd.read_csv("%s.csv" % "../../../../starters/allStarters")
# POSITION_PATH = '../../../../data/positionData/'

# buildPlayerGrades(position, source, df, sdf, POSITION_PATH, './')

# position = 'QB'
# source = pd.read_csv("%s.csv" % "../source/new_source")
# sdf = pd.read_csv("%s.csv" % "../../../../data/starters_23/starters_w1")

# df = buildNewPlayerGrades(position, source, sdf, './')