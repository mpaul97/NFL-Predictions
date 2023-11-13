import pandas as pd
import numpy as np
import os
import pickle
import regex as re
import statsmodels.api as sm
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import NuSVC, NuSVR, LinearSVC, LinearSVR

class GridSearch:
    def __init__(self, _dir):
        self._dir = _dir
        self.train = pd.read_csv("%s.csv" % (self._dir + "../train"))
        self.target = pd.read_csv("%s.csv" % (self._dir + "../target"))
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        self.models = {
            'lsvr': {
                'model': LinearSVR(),
                'params': {
                    'max_iter': [1000],
                    'C': [0.001],
                    'loss': ['squared_epsilon_insensitive']
                }
            }
        }
        return
    def find(self, targetName: str, modelName: str):
        X = self.train.drop(columns=self.str_cols)
        y = self.target[targetName]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model: LinearRegression = self.models[modelName]['model']
        params = self.models[modelName]['params']
        clf = GridSearchCV(model, params, verbose=2, n_jobs=-1)
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        return
    def evaluate(self, targetName: str):
        X = self.train.drop(columns=self.str_cols)
        y = self.target[targetName]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for modelName in self.models:
            model: LinearRegression = self.models[modelName]['model']
            params = self.models[modelName]['params']
            params = { p: params[p][0] for p in params }
            model.set_params(**params)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"Accuracy: {acc}")
        return
    
# END / GridSearch

########################

gs = GridSearch("./")

# gs.find('home_points', 'lsvr')

gs.evaluate('home_points')