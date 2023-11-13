import pandas as pd
import numpy as np
import os
import pickle
import regex as re
import statsmodels.api as sm
from ordered_set import OrderedSet
import multiprocessing
from itertools import repeat
from functools import partial, reduce
import joblib
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import NuSVC, NuSVR, LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

pd.options.mode.chained_assignment = None

class StatTest:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "../"
        self.str_cols = ['key', 'abbr', 'p_id', 'wy', 'position']
        return
    def testModel(self, position: str, target_name: str):
        train = pd.read_csv("%s.csv" % (self.data_dir + "train"))
        target = pd.read_csv("%s.csv" % (self.data_dir + "pred_targets"))
        train = train.loc[train['position']==position]
        target = target.loc[target['position']==position]
        target = target[self.str_cols+[target_name]]
        data = train.merge(target, on=self.str_cols)
        X = data.drop(columns=self.str_cols+[target_name])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        class_weight_dict = { 0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.8, 5: 0.4, 6: 0.3, 7: 0.1}
        model = LogisticRegression(n_jobs=-1, class_weight=class_weight_dict)
        # model = LinearSVC(class_weight=class_weight_dict)
        # model = LinearSVR()
        # model = RandomForestRegressor(n_jobs=-1)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f'Accuracy: {acc}')
        preds = model.predict(X_test)
        for i in range(10):
            p = preds[i]
            e = y_test.iloc[i]
            print(f"P: {p} - E: {e}")
        return
    
# END / StatTest

########################

st = StatTest("./")

st.testModel('QB', 'passing_touchdowns')