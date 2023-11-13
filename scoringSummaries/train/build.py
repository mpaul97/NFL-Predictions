import pandas as pd
import numpy as np
import os
import pickle

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

pd.options.mode.chained_assignment = None

def buildInfo(keys):
    df = pd.read_csv("%s.csv" % "../all/info")
    df = df.loc[df['key'].isin(keys)]
    df.to_csv("%s.csv" % "info", index=False)
    return

def buildInfoTargetFromOld():
    df = pd.read_csv("%s.csv" % "info")
    tdf = pd.read_csv("%s.csv" % "../../textEncoding/scoringSummaries/train/merged/targets_merged")
    tdf = tdf[['key', 'num', 'isDefensiveTouchdown', 'is2ptAttempt']]
    df = df.merge(tdf, on=['key', 'num'])
    df.to_csv("%s.csv" % "infoTarget", index=False)
    return

def addNewTarget(keyword, targetName):
    df = pd.read_csv("%s.csv" % "infoTarget")
    targets = []
    for index, row in df.iterrows():
        detail = row['detail']
        if keyword in detail:
            print(detail)
            val = input('Enter value - ' + targetName + ': ')
        else:
            val = 0
        targets.append(val)
    df[targetName] = targets
    df.to_csv("%s.csv" % "infoTarget", index=False)
    return

def getLines(df):
    lines = []
    # convert |info| to position
    for index, row in df.iterrows():
        line = row['detail']
        line_arr = line.split(" ")
        for i, word in enumerate(line_arr):
            if '|' in word:
                word_arr = word.split('|')
                pos = word_arr[-2]
                line_arr[i] = pos
            if word.isdigit() or word == 'unknown':
                line_arr[i] = 'number'
        line_arr = [l for l in line_arr if len(l) != 0]
        new_line = ' '.join(line_arr)
        lines.append(new_line)
    return lines

def vectorize():
    all_df = pd.read_csv("%s.csv" % "../all/info")
    df = pd.read_csv("%s.csv" % "../train/infoTarget")
    lines = getLines(df)
    # vectorize
    if 'cv.pickle' not in os.listdir('../models/'):
        all_lines = getLines(all_df)
        cv = CountVectorizer()
        X = cv.fit_transform(all_lines)
        # df to vector
        new_df1 = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
        new_df1.insert(0, 'key', all_df['key'].values)
        new_df1.insert(1, 'num', all_df['num'].values)
        new_df1.to_csv("%s.csv" % "../all/vector", index=False)
        pickle.dump(cv, open('../models/cv.pickle', 'wb'))
    else:
        cv = pickle.load(open('../models/cv.pickle', 'rb'))
    x = cv.transform(lines)
    # df to vector
    new_df = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())
    new_df.insert(0, 'key', df['key'].values)
    new_df.insert(1, 'num', df['num'].values)
    new_df.to_csv("%s.csv" % "vector", index=False)
    return

def addExtraFeatures():
    # features
    #   - line length
    #   - 'kick' in parantheses
    new_df = pd.DataFrame(columns=['key', 'num', 'lineLengths', 'isKickInParantheses'])
    # -------------------------
    df = pd.read_csv("%s.csv" % "vector")
    idf = pd.read_csv("%s.csv" % "infoTarget")
    for index, row in idf.iterrows():
        key = row['key']
        num = row['num']
        line = row['detail']
        line_length = len(line)
        # parantheses check
        isKick = 0
        if '(' in line:
            p_line = line[line.index('('):]
            if 'kick' in p_line:
                isKick = 1
        new_df.loc[len(new_df.index)] = [key, num, line_length, isKick]
    # merge with vector
    df = df.merge(new_df, on=['key', 'num'])
    df.to_csv("%s.csv" % "vector_extra", index=False)
    return

def saveModels(extra):
    idf = pd.read_csv("%s.csv" % "info")
    df = pd.read_csv("%s.csv" % "infoTarget")
    vdf = pd.read_csv("%s.csv" % "vector")
    if extra:
        vdf = pd.read_csv("%s.csv" % "vector_extra")
    # predict
    cols = set(df.columns).difference(set(idf.columns))
    x = vdf.drop(columns=['key', 'num'])
    for target in cols:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        preds = model.predict(X_test)
        # test true accuracy - predicted vs. expected
        print('------------------')
        count = 0
        for i in range(len(preds)):
            pred = preds[i]
            expected = y_test.values[i]
            expected_index = y_test.index[i]
            if pred != expected:
                print(str(pred) + " - " + str(expected))
                print(str(expected_index) + ': ' + df.iloc[expected_index]['detail'])
                count += 1
        print(target + ': ' + str(count) + '/' + str(len(preds)))
        print('Accuracy: ' + str(round(acc, 2)))
        print()
        pickle.dump(model, open('../models/' + target + '.sav', 'wb'))
    return

def predictAll():
    df = pd.read_csv("%s.csv" % "../all/vector")
    idf = pd.read_csv("%s.csv" % "../all/info")
    X = df.drop(columns=['key', 'num'])
    for fn in os.listdir('../models'):
        if '.sav' in fn:
            targetName = fn.replace('.sav','')
            model = pickle.load(open('../models/' + fn, 'rb'))
            idf[targetName] = model.predict(X)
    idf.to_csv('%s.csv' % '../all/infoTarget', index=False)
    return

#######################

keys = [
    '197911050mia', '200112090ram', '200212080was',
    '200412050chi', '200601010nyj', '201111060htx',
    '201212020det', '201312220sdg', '201901060chi',
    '202110030nyj', '202211130buf', '202212110sdg',
    '202301010gnb', '202301150buf', '202301150cin',
    '202301160tam', '202209110mia', '202211060nwe',
    '202211200nwe', '202212120crd', '202212170min',
    '199911280rai', '200410170jax', '200410170dal',
    '201411020kan', '201509200cle', '202301080clt'
]

# buildInfo(keys)

# buildInfoTargetFromOld()

# addNewTarget(
#     keyword='return',
#     targetName='isSpecialTeamsTouchdown'
# )

# vectorize()

# addExtraFeatures()

# saveModels(
#     extra=False
# )

predictAll()