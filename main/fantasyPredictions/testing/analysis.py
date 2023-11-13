import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

STR_COLS = ['key', 'abbr', 'p_id', 'wy', 'position']

def findOutliers():
    train = pd.read_csv("%s.csv" % "../train")
    target = pd.read_csv("%s.csv" % "../target")
    data: pd.DataFrame = train.merge(target, on=STR_COLS)
    print(data.columns)
    return

def testSeperatePositions():
    train = pd.read_csv("%s.csv" % "../train")
    target = pd.read_csv("%s.csv" % "../target")
    X = train.drop(columns=STR_COLS)
    y = target['points']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Normal accuracy: {acc}")
    positions = list(set(train['position'].values))
    for pos in positions:
        df: pd.DataFrame = train.loc[train['position']==pos]
        df = df.merge(target, on=STR_COLS)
        X1 = df.drop(columns=STR_COLS+['points', 'week_rank'])
        y1 = df['points']
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)
        model = LinearRegression(n_jobs=-1)
        model.fit(X1_train, y1_train)
        acc = model.score(X1_test, y1_test)
        print(f"Position ({pos}) accuracy: {acc}")
    return

######################

# findOutliers()

testSeperatePositions()