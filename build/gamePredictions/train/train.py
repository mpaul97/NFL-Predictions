from matplotlib.pyplot import grid
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LassoLars
from sklearn.svm import NuSVR
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFromModel

import statsmodels.api as sm
from pprint import pprint

pd.options.mode.chained_assignment = None

TARGET_PATH = "../targets/"

def getFeaturesAndLabels(trainNum, targetName):
    
    features = pd.read_csv("%s.csv" % ("train" + str(trainNum)))
    
    labels = pd.read_csv("%s.csv" % (TARGET_PATH + "target"))
    keep_cols = ['key', 'opp_abbr', 'wy', targetName]
    drop_cols = list(set(labels.columns).difference(set(keep_cols)))
    labels.drop(columns=drop_cols, inplace=True)
    
    return features, labels

def shortenData(features, labels, startYear):
    
    start = features.loc[features['wy'].str.contains(str(startYear))].index.values[0]
    
    features = features.loc[features.index>=start]
    labels = labels.loc[labels.index>=start]
    
    return features, labels

def olsStats(x, y, threshold, saveOls):
    
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()

    print(results.summary())

    pvals = results.pvalues;

    drop_cols = []

    for index, value in pvals.items():
        if value >= threshold and index != 'const':
            drop_cols.append(index)
            
    if saveOls:
        temp = results.pvalues.sort_values()
        temp.to_frame().to_csv("olsStats1.csv")

    return drop_cols

def forestFeatureSelection(x, y):
    
    y = y.values.ravel()
    
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(x, y)
    
    selected_features = list(x.columns[(sel.get_support())])
    
    drop_cols = list(set(set(x.columns).difference(set(selected_features))))
    
    return drop_cols

def alterData(features, labels, shorten, shortenYear, ols, olsThreshold, saveOls, fs):
    
    if shorten:
        features, labels = shortenData(features, labels, shortenYear)

    col_drops = ['key', 'opp_abbr', 'wy']

    x = features.drop(columns=col_drops)
    y = labels.drop(columns=col_drops)

    if ols and not fs:
        pDrops = olsStats(x, y, threshold=olsThreshold, saveOls=saveOls)
        print("pDrops:", pDrops)
        features.drop(columns=pDrops, inplace=True)

    if fs and not ols:
        fDrops = forestFeatureSelection(x, y)
        print("fDrops:", fDrops)
        features.drop(columns=fDrops, inplace=True)
        
    if ols and fs:
        print("Alter Data Error: ols and fs both True!")
        
    return features, labels  

def gridTuning(features, labels, modelName):
        
    # x_scaler, y_scaler = preprocessing.StandardScaler(), preprocessing.StandardScaler()
    # x_train = x_scaler.fit_transform(x_train)
    # y_train = y_scaler.fit_transform(y_train)
        
    if modelName == 'log':
        
        params = {
            'max_iter': [300, 400],
            'verbose': [0, 1, 2],
            'C': [0.25, 0.5, 0.75, 1.0]
        }
        
        model = LogisticRegression()
        
    # end models
        
    col_drops = ['key', 'opp_abbr', 'wy']
        
    features.drop(columns=col_drops, inplace=True)
    labels.drop(columns=col_drops, inplace=True)
    
    labels = labels.values.ravel()
        
    gridModel = GridSearchCV(model, param_grid=params, cv=10, verbose=0, n_jobs=-1)
    gridModel.fit(features, labels)
    
    print(gridModel.best_params_)
    print("---------------------------------")
    pprint(gridModel.cv_results_)
    
    return

def predict(features, labels, modelName, trainTest, targetName):
    
    if trainTest:
        
        x_train, x_test, y_train, y_test = train_test_split(features, 
                                                            labels, 
                                                            test_size=0.05, 
                                                            random_state=42
                                                            )
        
        testXCopy = x_test.copy()
        testYCopy = y_test.copy()
        
        col_drops = ['key', 'opp_abbr', 'wy']
        
        x_train.drop(columns=col_drops, inplace=True)
        y_train.drop(columns=col_drops, inplace=True)
        x_test.drop(columns=col_drops, inplace=True)
        y_test.drop(columns=col_drops, inplace=True)
        
    else:
    
        x_train = features.loc[features['wy']!='20 | 2021']
        y_train = labels.loc[labels['wy']!='20 | 2021']
        
        x_test = features.loc[features['wy']=='20 | 2021']
        y_test = labels.loc[labels['wy']=='20 | 2021']
        
        testXCopy = x_test.copy()
        testYCopy = y_test.copy()
        
        col_drops = ['key', 'opp_abbr', 'wy']
        
        x_train.drop(columns=col_drops, inplace=True)
        y_train.drop(columns=col_drops, inplace=True)
        x_test.drop(columns=col_drops, inplace=True)
        y_test.drop(columns=col_drops, inplace=True)
    
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    
    if modelName == 'log':
        model = LogisticRegression(
                    max_iter=500,
                    verbose=0,
                    random_state=42,
                    n_jobs=-1
                )
        y_train = y_train.values.ravel()
    elif modelName == 'forest':
        model = RandomForestClassifier(
                    n_estimators=800,
                    max_depth=24,
                    min_samples_split=12,
                    random_state=42, 
                    n_jobs=-1
                )
        y_train = y_train.values.ravel()
    elif modelName == 'forestReg':
        model = RandomForestRegressor(
                    n_estimators=500,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
        y_train = y_train.values.ravel()
    elif modelName == 'lasso':
        model = LassoLars(
                    alpha=0.001, 
                    max_iter=1000, 
                    random_state=42
                )
    elif modelName == 'nu':
        model = NuSVR(
                    nu=0.3, 
                    gamma='auto', 
                    cache_size=2000
                )
        y_train = y_train.values.ravel()
    elif modelName == 'svr':
        model = SVR(
                    C=1.0,
                    epsilon=0.2,
                    cache_size=2000
                )
        y_train = y_train.values.ravel()
    elif modelName == 'knn':
        model = KNeighborsClassifier(
                    n_neighbors=8
                )
        y_train = y_train.values.ravel()
    elif modelName == 'knnReg':
        model = KNeighborsRegressor(
                    n_neighbors=12    
                )
        y_train = y_train.values.ravel()
        
    model.fit(x_train, y_train)
    y_pred_log = model.predict(x_test)

    regModels = ['forestReg', 'lasso', 'nu', 'svr', 'knnReg']

    if modelName not in regModels:
        acc = metrics.accuracy_score(y_test, y_pred_log)
    else:
        acc = model.score(x_test, y_test)

    predictions = []

    for i in range(len(testXCopy.index)):
        name = testXCopy.iloc[i]['key']
        actualY = testYCopy.iloc[i][targetName]
        predictions.append((name, round(y_pred_log[i], 2), actualY))
        
    # predictions.sort(key=lambda x: x[0], reverse=True)
    
    predictions = predictions[:10]

    for p in predictions:
        print(p[0] + " : " + str(p[1]) + " <-> Actual: " + str(p[2]))

    print("Accuracy:", acc)

    return
    
######################################

targetName = 'points'

features, labels = getFeaturesAndLabels(14, targetName)

features, labels = alterData(
    features, 
    labels, 
    shorten=True,
    shortenYear=2005,
    ols=False,
    olsThreshold=0.5,
    saveOls=False,
    fs=False
)

# gridTuning(features, labels, 'log')

predict(features, 
        labels, 
        'forest',
        trainTest=True,
        targetName=targetName
)