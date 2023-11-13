import pandas as pd
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
import statsmodels.api as sm
from random import randrange

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
    
# ------------------------------------

# predict
def predictPerGameGrades(position, wys, _dir):
    
    train_dir = 'perGameTrain/'
    target_dir = 'perGameTargets/'
    
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
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)

        all_models.append((acc, model))
        
    all_models.sort(key=lambda x: x[0], reverse=True)
    
    best_model: LinearRegression = all_models[0][1]
    print('Accuracy:', all_models[0][0])
    
    z_test = test.drop(columns=['p_id', 'wy', 'game_key', 'abbr'])
    
    preds = model.predict(z_test)
    
    test['grade'] = preds
    test = test[['p_id', 'wy', 'game_key', 'abbr', 'grade']]
    test = test.round(4)
    
    test.to_csv("%s.csv" % (_dir + position + "_perGame_grades"), index=False)
    
    return

def getGrade(key, abbr, wy, df: pd.DataFrame, cd: pd.DataFrame):
    pids_info = cd.loc[(cd['game_key']==key)&(cd['abbr']==abbr), ['p_id', 'attempted_passes']].values
    if pids_info.shape[0] == 0: # no pids found
        return 100
    elif pids_info.shape[0] == 1:
        pid = pids_info[0][0]
    else:
        pids_info = pids_info[pids_info[:, 1].argsort()[::-1]]
        pid = pids_info[0][0]
    pdf: pd.DataFrame = df.loc[df['p_id']==pid]
    pdf.reset_index(drop=True, inplace=True)
    try:
        curr_index = pdf.loc[pdf['wy']==wy].index.values[0]
        grade = pdf.iloc[curr_index-1]['grade']
    except IndexError as error: # start of data
        grade = 100
    return pid, grade

def mergeGrades(position, df: pd.DataFrame, source: pd.DataFrame, cd: pd.DataFrame):
    home_grades, home_pids, away_grades, away_pids = [], [], [], []
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_pid, home_grade = getGrade(key, home_abbr, wy, df, cd)
        away_pid, away_grade = getGrade(key, away_abbr, wy, df, cd)
        home_grades.append(home_grade)
        home_pids.append(home_pid)
        away_grades.append(away_grade)
        away_pids.append(away_pid)
    source['home_grade_' + position] = home_grades
    source['away_grade_' + position] = away_grades
    return source

# build each teams rankings per week
def buildPlayerGrades(position, source: pd.DataFrame, df: pd.DataFrame, POSITION_PATH, _dir):
    
    # PREDICT PERGAME EVERY RUN - CD WILL HAVE NEW WEEK DATA/MOST RECENT PERGAME GRADES FOR NEW WEEK
    predictPerGameGrades('QB', ['5 | 2022', '6 | 2022', '7 | 2022'], _dir)
    
    # if (position + '_playerGrades.csv') in os.listdir(_dir):
    #     print(position + '_playerGrades already created.')
    #     return
    
    print('Creating ' + position + '_playerGrades...')
    
    cd = pd.read_csv("%s.csv" % (POSITION_PATH + position + "Data"))
    gdf = pd.read_csv("%s.csv" % (_dir + position + '_perGame_grades'))
    
    cd = cd.loc[cd['game_key'].isin(df['key'].values)]
    all_pids = list(set(cd['p_id'].values))
    
    new_df = pd.DataFrame(columns=['p_id', 'wy', 'grade'])
    
    mean = np.mean(gdf['grade'].values)
    
    for pid in all_pids:
        # print(pid)
        g = Grade(pid, 100)
        data = gdf.loc[gdf['p_id']==pid]
        data.reset_index(drop=True, inplace=True)
        for index, row in data.iterrows():
            wy = row['wy']
            year = int(wy.split(" | ")[1])
            if index != len(data.index) - 1:
                next_wy = data.iloc[index+1]['wy']
                next_year = int(next_wy.split(" | ")[1])
                if year != next_year: # new season
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
            
    source = mergeGrades(position, new_df, source, cd)
    
    source.to_csv("%s.csv" % (_dir + position + "_playerGrades"), index=False)
    
    return

def buildNewPlayerGrades(position, source: pd.DataFrame, sdf: pd.DataFrame, _dir):
    
    gdf = pd.read_csv("%s.csv" % (_dir + position + '_perGame_grades'))
    df = pd.read_csv("%s.csv" % (_dir + position + '_playerGrades'))
    
    wy = source['wy'].values[0]
    week = int(wy.split(" | ")[0])
    
    home_grades, away_grades = [], []
    
    # if week == 1:
    for index, row in source.iterrows():
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_starters = (sdf.loc[sdf['abbr']==home_abbr, 'starters'].values[0]).split("|")
        away_starters = (sdf.loc[sdf['abbr']==away_abbr, 'starters'].values[0]).split("|")
        home_pids = [s.replace((':' + position), '') for s in home_starters if position in s]
        away_pids = [s.replace((':' + position), '') for s in away_starters if position in s]
        for pid in home_pids:
            grade = df.loc[df['p_id']==pid]
            print(grade)
    
    return

###################################

# position = 'QB'
# source = pd.read_csv("%s.csv" % "../source/new_source")
# sdf = pd.read_csv("%s.csv" % "../../../../data/starters/starters_w22")

# buildNewPlayerGrades(position, source, sdf, './')

