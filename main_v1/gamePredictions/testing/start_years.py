import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

STR_COLS = ['key', 'wy', 'home_abbr', 'away_abbr']

def getDrops(X, y, threshold):
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    
    pdf: pd.Series = results.pvalues.sort_values()
    ols = pdf.to_frame()
    ols.insert(0, 'name', ols.index)
    ols.columns = ['name', 'val']
    ols.fillna(1, inplace=True)
    
    drops = []

    for index, row in ols.iterrows():
        name = row['name']
        val = row['val']
        if val > threshold and name != 'const':
            drops.append(name)

    return drops

def test_year(year: int):
    train = pd.read_csv("%s.csv" % "../train")
    target = pd.read_csv("%s.csv" % "../target")
    target = target[STR_COLS+['home_won']]
    data = train.merge(target, on=STR_COLS)
    start = data.loc[data['wy'].str.contains(str(year))].index.values[0]
    data: pd.DataFrame = data.loc[data.index>=start]
    data = data.reset_index()
    # predict
    X = data.drop(columns=STR_COLS+['home_won'])
    y = data['home_won']
    X = X.drop(columns=getDrops(X, y, 0.2))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return round(acc, 2)

def test_all():
    accs = {}
    for year in range(2000, 2016):
        accs[year] = test_year(year)
    print(accs)
    return

#################

test_all()

