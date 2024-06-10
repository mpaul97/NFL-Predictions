import pandas as pd
import numpy as np
import os
import pickle
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

STR_COLS = ['key', 'abbr', 'p_id', 'wy', 'position']
OLS_THRESHOLD = 0.1

# get one hot encoding for feature
def toOneHotEncoding(col, df: pd.DataFrame):
    vals = df[col]
    lb = LabelEncoder()
    lb_vals = lb.fit_transform(vals)
    ohe = OneHotEncoder()
    ohe_vals = ohe.fit_transform(lb_vals.reshape(-1, 1)).toarray()
    cols = [col + '_' + str(i) for i in range(ohe_vals.shape[1])]
    new_df = pd.DataFrame(ohe_vals, columns=cols)
    df = pd.concat([df, new_df], axis=1)
    df.drop(columns=[col], inplace=True)
    return df

# get label encoding for feature
def toLabelEncoding(col, df: pd.DataFrame):
    lb = LabelEncoder()
    df[col] = lb.fit_transform(df[col])
    return df

# save ols stats
def saveOlsStats(X, y, _type, _dir):
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    
    pdf: pd.Series = results.pvalues.sort_values()
    ols = pdf.to_frame()
    ols.insert(0, 'name', ols.index)
    ols.columns = ['name', 'val']
    ols.fillna(1, inplace=True)
    ols.to_csv("%s.csv" % (_dir + 'testing/olsStats_' + _type), index=False)
    
    drops = []

    for index, row in ols.iterrows():
        name = row['name']
        val = row['val']
        if val > OLS_THRESHOLD and name != 'const':
            drops.append(name)

    return drops

# get ols drops
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

# predict
def predict(models, df: pd.DataFrame, y, _dir, _type):
    X = df.drop(STR_COLS, axis=1)
    drops = saveOlsStats(X, y, _type, _dir)
    X = X.drop(drops, axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # model = LogisticRegression(
        #     n_jobs=-1
        # )
    for modelName, model in models:
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(('Accuracy - ' + modelName + ':'), acc)
    return

# test models - encode/drops etc.
def testModels(models, targetName, _dir):
    position = 'QB'
    df = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target = target[STR_COLS+[targetName]]
    df = df.merge(target, on=STR_COLS)
    df = df.loc[df['position']==position]
    predict(models, df.drop(columns=[targetName]), df[targetName], _dir, 'original')
    # df = toOneHotEncoding('home_season', df)
    # df = toOneHotEncoding('away_season', df)
    # print('New:')
    # predict(df, _dir, 'new')
    return

# get accuracy for ols threshold testing - suppress convergencewarning/max_iter
def getAccuracy(X: pd.DataFrame, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return acc

# find best ols thresholds
def findBestThresholds(_dir):
    file = open('best_thresholds.txt', 'w')
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target = target.round(0)
    target_cols = list(set(target.columns).difference(set(STR_COLS)))
    for col in target_cols:
        info = []
        for i in range(1, 10):
            print(col, str(i))
            threshold = i/10
            X = train.drop(STR_COLS, axis=1)
            y = target[col]
            drops = getDrops(X, y, threshold)
            X1 = X.drop(columns=drops)
            acc = getAccuracy(X1, y)
            info.append((acc, threshold))
        info.sort(key=lambda x: x[0], reverse=True)
        # print(col + " -> Best threshold: " + str(info[0][1]))
        file.write(col + " -> Best threshold: " + str(info[0][1]) + " : " + str(info[0][0]) + "\n")

    return

# get heatmap
def getCorrHeatMap(_dir):
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    # cols = list(set(train.columns).difference(set(STR_COLS)))
    # chunks = np.array_split(cols, 15)
    # train = train[STR_COLS+list(chunks[0])]
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target = target[STR_COLS+['points']]
    data: pd.DataFrame = train.merge(target, on=STR_COLS)
    corrmat = data.corr()
    k = 10
    cols = corrmat.nlargest(k, 'points')['points'].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return

# get outliers
def outliers(_dir):
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target = target[STR_COLS+['points']]
    new_target = target.loc[target['points'].between(0, 50)]
    data: pd.DataFrame = new_target.merge(train, on=STR_COLS)
    n_acc = getAccuracy(train.drop(columns=STR_COLS), target.drop(columns=STR_COLS))
    print(f"Normal accuracy: {n_acc}")
    o_acc = getAccuracy(data.drop(columns=STR_COLS+['points']), data['points'])
    print(f"No outlier accuracy: {o_acc}")
    return

# test point range predictions
def point_ranges(_dir):
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target['over_25'] = target['points'].apply(lambda x: x >= 25).astype(int)
    target['over_20'] = target['points'].apply(lambda x: x >= 20).astype(int)
    target['over_15'] = target['points'].apply(lambda x: x >= 15).astype(int)
    target['over_10'] = target['points'].apply(lambda x: x >= 10).astype(int)
    target['over_5'] = target['points'].apply(lambda x: x >= 5).astype(int)
    target['under_5'] = target['points'].apply(lambda x: x < 5).astype(int)
    df = train.merge(target, on=STR_COLS)
    t_cols = ['over_25', 'over_20', 'over_15','over_10', 'over_5', 'under_5']
    X = df.drop(columns=STR_COLS+['points', 'week_rank']+t_cols)
    for col in t_cols[:1]:
        y = df[col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(n_jobs=-1)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"{col} - accuracy: {acc}")
    return

# add point ranges to target
def add_point_ranges(_dir):
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target['over_25'] = target['points'].apply(lambda x: x >= 25).astype(int)
    target['over_20'] = target['points'].apply(lambda x: x >= 20).astype(int)
    target['over_15'] = target['points'].apply(lambda x: x >= 15).astype(int)
    target['over_10'] = target['points'].apply(lambda x: x >= 10).astype(int)
    target['over_5'] = target['points'].apply(lambda x: x >= 5).astype(int)
    target['under_5'] = target['points'].apply(lambda x: x < 5).astype(int)
    target.to_csv("%s.csv" % (_dir + "target"), index=False)
    return

###############################

# find best model
models = [
    ('lir', LinearRegression())
    # ('log', LogisticRegression())
]
testModels(models, 'points', '../')

# findBestThresholds('../')

# getCorrHeatMap('../')

# outliers('../')

# point_ranges('../')

# add_point_ranges('../')