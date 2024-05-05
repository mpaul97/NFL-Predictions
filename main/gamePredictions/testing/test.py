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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

STR_COLS = ['key', 'wy', 'home_abbr', 'away_abbr']
OLS_THRESHOLDS = {'away_points': 0.4, 'home_points': 0.1, 'home_won': 0.1}

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
def saveOlsStats(X, y, targetName, _type, _dir):
    
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
        if val > OLS_THRESHOLDS[targetName] and name != 'const':
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
def predict(df: pd.DataFrame, y, modelName, targetName, _dir, _type):
    X = df.drop(STR_COLS, axis=1)
    drops = saveOlsStats(X, y, targetName, _type, _dir)
    X = X.drop(drops, axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if modelName == 'log':
        model = LogisticRegression(
            n_jobs=-1
        )
    elif modelName == 'forest':
        model = RandomForestClassifier(
            n_jobs=-1
        )
    elif modelName == 'knn':
        model = KNeighborsClassifier(
            n_jobs=-1
        )
    elif modelName == 'linear':
        model = LinearRegression(
            n_jobs=-1
        )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print((modelName + ' - accuracy:'), acc)
    return

# test models - ols
def testModels(modelNames, targetName, _dir):
    df = pd.read_csv("%s.csv" % (_dir + "train_short"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    df = df.merge(target, on=STR_COLS)
    target = df[STR_COLS+['home_won', 'home_points', 'away_points']]
    df.drop(columns=['home_won', 'home_points', 'away_points'], inplace=True)
    y = target[targetName]
    for modelName in modelNames:
        predict(df, y, modelName, targetName, _dir, '')
    return

# test models - one hot encoding
def testModelsEncoding(targetName, _dir):
    df = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    y = target[targetName]
    print('Original:')
    predict(df, y, 'log', targetName, _dir, 'original')
    df = toOneHotEncoding('home_pass_touchdowns_rank', df)
    df = toOneHotEncoding('away_pass_touchdowns_rank', df)
    print('New:')
    predict(df, y, 'log', targetName, _dir, 'new')
    return

# get accuracy for ols threshold testing - suppress convergencewarning/max_iter
def getAccuracy(X: pd.DataFrame, y, col):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression() if 'points' not in col else RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return acc

# get best ols drop thresholds for each target
def findBestThresholds(_dir):
    file = open('best_thresholds.txt', 'w')
    train = pd.read_csv("%s.csv" % (_dir + "train_short"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target_cols = list(set(target.columns).difference(set(STR_COLS)))
    df = train.merge(target, on=STR_COLS)
    train = df.drop(columns=target_cols)
    target = df[STR_COLS+target_cols]
    for col in target_cols:
        info = []
        for i in range(1, 10):
            print(col, str(i))
            threshold = i/20
            X = train.drop(STR_COLS, axis=1)
            y = target[col]
            drops = getDrops(X, y, threshold)
            X1 = X.drop(columns=drops)
            acc = getAccuracy(X1, y, col)
            info.append((acc, threshold))
        info.sort(key=lambda x: x[0], reverse=True)
        # print(col + " -> Best threshold: " + str(info[0][1]))
        file.write(col + " -> Best threshold: " + str(info[0][1]) + " : " + str(info[0][0]) + "\n")
    return

# get incorrect predictions
def getIncorrectPredictions(targetName, rebuild, _dir):
    df = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    if rebuild:
        y = target[targetName]
        X = df.drop(STR_COLS, axis=1)
        drops = getDrops(X, y, threshold=OLS_THRESHOLDS[targetName])
        X = X.drop(drops, axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print('Accuracy:', acc)
        preds = model.predict(X_test)
        t_test = target.loc[target.index.isin(y_test.index.values)]
        t_test = t_test[STR_COLS+[targetName]]
        t_test[targetName + '_pred'] = preds
        mask = t_test[targetName] != t_test[targetName + '_pred'] # get incorrect preds
        t_test: pd.DataFrame = t_test[mask]
        t_test.to_csv("%s.csv" % "incorrectPreds", index=False)
    else: # visualize incorrect train attributes
        t_test = pd.read_csv("%s.csv" % "incorrectPreds")
        i_train = t_test.merge(df, on=STR_COLS)
        # i_train.to_csv("%s.csv" % "incorrectPreds_train", index=False)
        i_train = i_train.loc[i_train['wy'].str.contains('2022')]
        pred_home_won: pd.DataFrame = i_train.loc[i_train[targetName+'_pred']==1]
        # pred_home_lost = i_train.loc[i_train[targetName+'_pred']==0]
        cols = ['division_standings', 'conference_standings', 'elo']
        test_cols = [_type + '_' + col for col in cols for _type in ['home', 'away']]
        pred_home_won = pred_home_won[STR_COLS+[targetName, targetName+'_pred']+test_cols]
        for col in cols:
            if 'elo' in col:
                pred_home_won['home_' + col + '_better'] = pred_home_won['home_' + col] > pred_home_won['away_' + col]
            else:
                pred_home_won['home_' + col + '_better'] = pred_home_won['home_' + col] < pred_home_won['away_' + col]
        pred_home_won.to_csv("%s.csv" % "temp", index=False)
    return

# get heatmap
def getCorrHeatMap(_dir):
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    # cols = list(set(train.columns).difference(set(STR_COLS)))
    # chunks = np.array_split(cols, 15)
    # train = train[STR_COLS+list(chunks[0])]
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target = target[STR_COLS+['home_won']]
    data: pd.DataFrame = train.merge(target, on=STR_COLS)
    corrmat = data.corr()
    k = 20
    cols = corrmat.nlargest(k, 'home_won')['home_won'].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=0.75)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return

# test corr cols
def testCorrCols(_dir):
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    target = target[STR_COLS+['home_won']]
    data: pd.DataFrame = train.merge(target, on=STR_COLS)
    corrmat = data.corr()
    k = 50
    cols = list(corrmat.nlargest(k, 'home_won')['home_won'].index)
    cols.remove('home_won')
    X = data[cols]
    y = data['home_won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Accuracy: {acc}")
    # test = pd.read_csv("%s.csv" % (_dir + "test"))
    # test = test[cols]
    # preds = model.predict(test)
    # print(preds)
    return

# get multicollinearity(cols giving same info - highly correlated) columns
def getMultiCols(_dir):
    df = pd.read_csv("%s.csv" % (_dir + "train"))
    df.drop(columns=STR_COLS)
    corrmat = df.corr()
    corrmat = corrmat.head()
    cols = corrmat.columns
    for index, row in corrmat.iterrows():
        for col in cols:
            val = row[col]
            if val > 0.5 and index != col:
                print(index, col)
    return

# test logistic regression predict probabilites
def predict_probs(_dir):
    df = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    X = df.drop(STR_COLS, axis=1)
    y = target['home_won']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    preds = preds[:10, :]
    for i in range(preds.shape[0]):
        prob_0, prob_1 = preds[i]
        prob_0, prob_1 = round(prob_0, 2), round(prob_1, 2)
        print(f"Prob 0: {prob_0}, Prob 1: {prob_1} -> Actual: {y_test.iloc[i]}")
    return

def compare_train_test():
    train = pd.read_csv("%s.csv" % "../train")
    test = pd.read_csv("%s.csv" % "../test")
    nans = test.isna().any()
    for index, item in nans.items():
        if item:
            print(index)
    # for col in test.columns:
    #     if col not in train.columns:
    #         print(col)
    return

###############################

if __name__ == '__main__':
    # testModels(
    #     modelNames=['log'],
    #     targetName='home_won', 
    #     _dir='../'
    # )
    # getIncorrectPredictions(
    #     targetName='home_won',
    #     rebuild=False,
    #     _dir='../'
    # )
    # testModelsEncoding('home_won', '../')
    # findBestThresholds('../')
    # getCorrHeatMap('../')
    # testCorrCols('../')
    # getMultiCols('../')
    # predict_probs('../')
    compare_train_test()