import pandas as pd
import numpy as np
import os
import pickle
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# try:
#     from features.pred_standings.tiebreakerAttributes import buildTiebreakerAttributes
# except ModuleNotFoundError:
#     print('pred_standings - Using local imports.')
#     from tiebreakerAttributes import buildTiebreakerAttributes
from gamePredictions.features.pred_standings.tiebreakerAttributes import buildTiebreakerAttributes

def sourceToIndiv(source: pd.DataFrame):
    new_df = pd.DataFrame(columns=['key', 'abbr', 'opp_abbr', 'wy'])
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        new_df.loc[len(new_df.index)] = [key, home_abbr, away_abbr, wy]
        new_df.loc[len(new_df.index)] = [key, away_abbr, home_abbr, wy]
    return new_df

def encode(_dir):
    
    df = pd.concat([
        pd.read_csv("%s.csv" % (_dir + "rawTrain_0")),
        pd.read_csv("%s.csv" % (_dir + "rawTrain_1"))
    ])
        
    st = pd.read_csv("%s.csv" % (_dir + "tiebreakerAttributes"))
    
    m_dir = _dir + 'models/'
    
    div_encoder = LabelEncoder()
    div_encoder.fit(st['division'])
    df['division'] = div_encoder.transform(df['division'])
    np.save((m_dir + 'div_encoder.npy'), div_encoder.classes_)
    
    conf_encoder = LabelEncoder()
    conf_encoder.fit(st['conference'])
    df['conference'] = conf_encoder.transform(df['conference'])
    np.save((m_dir + 'conf_encoder.npy'), conf_encoder.classes_)
    
    df.to_csv("%s.csv" % (_dir + "train"), index=False)
    
    return

# save best models
def saveModels(_dir):
    
    data = pd.read_csv("%s.csv" % (_dir + 'train'))
    
    target_cols = ['division_standings', 'conference_standings']
    
    drop_cols = ['abbr', 'wy']
    
    m_dir = _dir + 'models/'
    
    best_accs = [0, 0]
    
    for i, col in enumerate(target_cols):
        while best_accs[i] < 0.7:
            x = data.drop(columns=drop_cols+target_cols)
            y = data[col]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
            model = RandomForestClassifier(n_jobs=-1)
            model.fit(x_train, y_train)
            acc = model.score(x_test, y_test)
            if acc > best_accs[i]:
                pickle.dump(model, open((m_dir + col + '.sav'), 'wb'))
                best_accs[i] = acc
                print('Accuracy - ' + col + ':', acc)
    
    return

def predict(df: pd.DataFrame, _dir):
    
    m_dir = _dir + 'models/'
    
    div_encoder = LabelEncoder()
    div_encoder.classes_ = np.load(m_dir + "div_encoder.npy", allow_pickle=True)
    df['division'] = div_encoder.transform(df['division'])
    
    conf_encoder = LabelEncoder()
    conf_encoder.classes_ = np.load(m_dir + "conf_encoder.npy", allow_pickle=True)
    df['conference'] = conf_encoder.transform(df['conference'])
    
    X = df.drop(columns=['abbr', 'wy'])
    
    target_cols = ['division_standings', 'conference_standings']
    
    new_df = pd.DataFrame()
    new_df['abbr'] = df['abbr']
    new_df['wy'] = df['wy']
    
    for col in target_cols:
        model: RandomForestClassifier = pickle.load(open((m_dir + col + '.sav'), 'rb'))
        preds = model.predict(X)
        new_df[col] = preds
    
    return new_df

def joinPreds(preds: pd.DataFrame, source: pd.DataFrame, _dir):
    
    target_cols = ['division_standings', 'conference_standings']
    
    home_cols = ['home_' + col for col in target_cols]
    away_cols = ['away_' + col for col in target_cols]
    
    new_df = pd.DataFrame(columns=list(source.columns)+home_cols+away_cols)
    
    for index, row in source.iterrows():
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        wy = row['wy']
        try:
            home_stats = preds.loc[(preds['wy']==wy)&(preds['abbr']==home_abbr)].values[0][-2:]
            away_stats = preds.loc[(preds['wy']==wy)&(preds['abbr']==away_abbr)].values[0][-2:]
        except IndexError:
            year = int(wy.split(" | ")[1])
            last_wy = preds.loc[preds['wy'].str.contains(str(year)), 'wy'].values[-1]
            home_stats = preds.loc[(preds['wy']==last_wy)&(preds['abbr']==home_abbr)].values[0][-2:]
            away_stats = preds.loc[(preds['wy']==last_wy)&(preds['abbr']==away_abbr)].values[0][-2:]
        new_df.loc[len(new_df.index)] = np.concatenate([row.values, home_stats, away_stats])
        
    new_df.to_csv("%s.csv" % (_dir + 'pred_standings'), index=False)
    
    return

def buildPredStandings(sourceIndiv: pd.DataFrame, source: pd.DataFrame, cd: pd.DataFrame, sl: pd.DataFrame, _dir):
    
    # build tiebreakerAttributes
    tb = buildTiebreakerAttributes(sourceIndiv, cd, sl, _dir)
    
    # encode conference and divisions, save encodings
    encode(_dir)

    # save models
    if 'division_standings.sav' not in os.listdir(_dir + 'models/'):
        print('Saving pred_standings models...')
        saveModels(_dir)
    else:
        print('pred_standings models already created.')
        
    # predict pred_standings
    preds = predict(tb, _dir)
    
    # join preds
    if 'pred_standings.csv' not in os.listdir(_dir):
        print('Creating pred_standings...')
        joinPreds(preds, source, _dir)
    else:
        print('pred_standings already created.')
    
    return

def buildNewPredStandings(source: pd.DataFrame, _dir):
    
    print('Creating new_pred_standings...')
    
    tb = pd.read_csv("%s.csv" % (_dir + 'tiebreakerAttributes'))
    
    # *****if wy in cd thats not in tb and wy is not playoffs*****
    # use most recent team attributes
    
    new_tb = pd.DataFrame(columns=tb.columns)
    
    for index, row in source.iterrows():
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        new_tb.loc[len(new_tb.index)] = tb.loc[tb['abbr']==home_abbr].values[-1]
        new_tb.loc[len(new_tb.index)] = tb.loc[tb['abbr']==away_abbr].values[-1]
    
    preds = predict(new_tb, _dir)
    
    # preds to source format
    target_cols = ['division_standings', 'conference_standings']
    
    home_cols = ['home_' + col for col in target_cols]
    away_cols = ['away_' + col for col in target_cols]
    
    new_df = pd.DataFrame(columns=list(source.columns)+home_cols+away_cols)
    
    for index, row in source.iterrows():
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_vals = preds.loc[preds['abbr']==home_abbr].values[0][-2:]
        away_vals = preds.loc[preds['abbr']==away_abbr].values[0][-2:]
        new_df.loc[len(new_df.index)] = np.concatenate([row.values, home_vals, away_vals])
    
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + 'newPred_standings'), index=False)
    
    return new_df

######################

# saveModels('./')

# preds = predict('./')
# source = pd.read_csv("%s.csv" % "../source/source")

# joinPreds(preds, source, './')

# sourceIndiv = pd.read_csv("%s.csv" % "../source/sourceIndividual")
# source = pd.read_csv("%s.csv" % "../source/new_source")

# buildNewPredStandings(source, './')