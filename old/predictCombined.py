from matplotlib.pyplot import grid
import pandas as pd
import numpy as np
import os
import random
import time
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
tf.function(experimental_relax_shapes=True)

from tensorflow import keras
from tensorflow.keras import layers

from PredictAttributes import pa as predict_attributes

GAME_TARGET_PATH = "../../gamePredictions/both/targets/"
GAME_TRAIN_PATH = "../../gamePredictions/both/train/"
GAME_TEST_PATH = "../../gamePredictions/both/predict/"

def getFeaturesLabelsTest(targetType, targetName, week, _type):
    
    files = os.listdir(".")
    feature_fn = [f for f in files if "trainCombined" in f and "csv" in f][0]
    
    if _type == 'tr':
        feature_fn = "g22.csv"
    
    features = pd.read_csv(feature_fn)
    labels = pd.read_csv("%s.csv" % (GAME_TARGET_PATH + "target"))
    
    files = os.listdir(".")
    test_fn = [f for f in files if "testCombined" in f and "csv" in f][0]
    
    if week == 3:
        test_fn = "test_w3.csv"
    
    if week == 20:
        test_fn = "test_w20.csv"
    
    if _type ==  'tr':
        test_fn = "g22_test.csv"

    test = pd.read_csv(test_fn)

    keep_cols = ['key', 'wy', targetName]
    drop_cols = list(set(labels.columns).difference(set(keep_cols)))
    labels.drop(columns=drop_cols, inplace=True)
    labels = labels.round(0)
    labels = features.merge(labels, on=['key', 'wy'], how='left')
    labels = labels[keep_cols]
    
    return features, labels, test

def getFeaturesIndividual(targetType, num):
    if targetType == 'g':
        features = pd.read_csv("%s.csv" % (GAME_TRAIN_PATH + "train" + str(num)))
    return features

def getTestDirAndNum(targetType, week, year):
    if targetType == 'g':
        test_dir = GAME_TEST_PATH + str(week) + "-" + str(year) + "/"
        testNum = max([int(fn.replace("test","").replace(".csv","")) for fn in os.listdir(test_dir) if 'test' in fn and '.csv' in fn])
    return test_dir, testNum

def getFeaturesTestIndividual(targetType, week, year):
    
    test_dir, num = getTestDirAndNum(targetType, week, year)

    test = pd.read_csv("%s.csv" % (test_dir + "test" + str(num)))

    features = getFeaturesIndividual(targetType, num)
    
    return features, test

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

    col_drops = ['key', 'home_abbr', 'away_abbr', 'wy']
    y_col_drops = ['key', 'wy']

    x = features.drop(columns=col_drops)
    y = labels.drop(columns=y_col_drops)

    drops = None

    if ols and not fs:
        pDrops = olsStats(x, y, threshold=olsThreshold, saveOls=saveOls)
        drops = pDrops
        # print("pDrops:", pDrops)
        features.drop(columns=pDrops, inplace=True)

    if fs and not ols:
        fDrops = forestFeatureSelection(x, y)
        drops = fDrops
        # print("fDrops:", fDrops)
        features.drop(columns=fDrops, inplace=True)
        
    if ols and fs:
        print("Alter Data Error: ols and fs both True!")
        
    return features, labels, drops

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

def getSamplingStrategy(sample_fs):
    
    sample_fs = {k: v for k, v in sorted(sample_fs.items(), key=lambda item: item[1], reverse=True)}

    max_val = list(sample_fs.values())[0] # max count labelmax_val = sample_fs[0] # max count label

    # total sum of other labels, not max
    other_total = sum([val for key, val in sample_fs.items() if val != max_val])

    sampling_strategy = {}

    for key, val in sample_fs.items():
        if val != max_val:
            new_val = int(round(((val/other_total)*(max_val+(other_total/2))), 0))
            if new_val > val: # not greater than original value
                if new_val < max_val: # not greater than max value
                    sampling_strategy[key] = new_val
                else:
                    sampling_strategy[key] = max_val
            else:
                sampling_strategy[key] = val
        else:
            sampling_strategy[key] = val
    
    return sampling_strategy

def sampleData(features, labels, targetName, oversample, strategy):
    # oversampling/undersampling for imbalanced labels (receiving_touchdowns, etc.)
    if oversample:
        sample_fs = Counter(labels[targetName].values.flatten())
        print(list(sample_fs.items())[:10])
        if strategy == 'dynamic':
            sampling_strategy = getSamplingStrategy(sample_fs)
        else:
            sampling_strategy = strategy
        oversample = RandomOverSampler(sampling_strategy=sampling_strategy)
        x_over, y_over = oversample.fit_resample(features, labels[targetName])
        print(list(Counter(y_over.values.flatten()).items())[:10])
        features = x_over
        labels = y_over
    return features, labels

def addPreds(preds, pred_name):
    new_df = pd.DataFrame(columns=['key', 'home_abbr', 'away_abbr', pred_name])
    for p in preds:
        new_df.loc[len(new_df.index)] = [p[0], p[1], p[2], p[3]]
    return new_df

def mostCommon(List):
    return max(set(List), key=List.count)

def predictSk(features, labels, test, modelName):
        
    x_train, x_test, y_train, y_test = train_test_split(features, 
                                                        labels, 
                                                        test_size=0.01, 
                                                        random_state=42
                                                        )
    
    col_drops = ['key', 'home_abbr', 'away_abbr', 'wy']
    y_col_drops = ['key', 'wy']
    
    testCopy = test.copy()
    test.drop(columns=col_drops, inplace=True)
    
    x_train.drop(columns=col_drops, inplace=True)
    y_train.drop(columns=y_col_drops, inplace=True)
    x_test.drop(columns=col_drops, inplace=True)
    y_test.drop(columns=y_col_drops, inplace=True)
    
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

    for i in range(len(testCopy.index)):
        name = testCopy.iloc[i]['key']
        home_abbr = testCopy.iloc[i]['home_abbr']
        away_abbr = testCopy.iloc[i]['away_abbr']
        predictions.append((name, home_abbr, away_abbr, round(y_pred_log[i], 2)))
        
    # predictions.sort(key=lambda x: x[0], reverse=True)

    # for p in predictions:
    #     print(p[0] + " Home:" + p[2] + " Away:" + p[3] + " : " + str(p[1]))
    
    print("Accuracy:", acc)

    return predictions, acc
    
def predict(features, labels, test, targetName, targetType, epochs, classFunc):
    
    x_train, x_test, y_train, y_test = train_test_split(features, 
                                                        labels, 
                                                        test_size=0.01, 
                                                        random_state=42
                                                        )
    
    col_drops = ['key', 'home_abbr', 'away_abbr', 'wy']
    y_col_drops = ['key', 'wy']
    
    testCopy = test.copy()
    test.drop(columns=col_drops, inplace=True)
    
    x_train.drop(columns=col_drops, inplace=True)
    y_train.drop(columns=y_col_drops, inplace=True)
    x_test.drop(columns=col_drops, inplace=True)
    y_test.drop(columns=y_col_drops, inplace=True)
    
    isClass = False
    # encode multiclass
    if 'touchdowns' in targetName or 'won' in targetName:
        isClass = True
        encoder = OneHotEncoder()
        if type(labels) == pd.Series:
            encoder.fit((labels.values).reshape(-1, 1))
        else:
            encoder.fit((labels[targetName].values).reshape(-1, 1))
        y_train = (encoder.transform((y_train.values).reshape(-1, 1))).toarray()
        y_test = (encoder.transform((y_test.values).reshape(-1, 1))).toarray()
        
    # normalize
    normalizer = tf.keras.layers.Normalization(input_shape=[len(x_train.columns), ], axis=-1)
    normalizer.adapt(np.array(x_train))

    normal_X_train = normalizer(x_train).numpy()
    normal_X_test = normalizer(x_test).numpy()
    
    if isClass:
        # class model
        # ---------------------------------------------------------------------------------
        # class nums for output dense layer
        final_dense_val = y_train.shape[1]
        
        # shape
        input_shape = (normal_X_train.shape[1],)

        model = tf.keras.Sequential([
            layers.Dense((final_dense_val**2), input_shape=input_shape, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense((final_dense_val*2), activation='relu'),
            layers.Dense((final_dense_val*2), activation='relu'),
            layers.Dense(final_dense_val, activation=classFunc)
        ])
        
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )
        
        tf.function(experimental_relax_shapes=True)
        model.fit(
            normal_X_train,
            y_train,
            verbose=2,
            epochs=epochs,
            validation_split=0.2
        )
    else:
        # regression model
        # ---------------------------------------------------------------------------------
        input_shape = (normal_X_train.shape[1],)

        model = tf.keras.Sequential([
            layers.Dense(500, input_shape=input_shape, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(100, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            loss=tf.keras.losses.mae,
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
            metrics=['accuracy']
        )

        tf.function(experimental_relax_shapes=True)
        model.fit(
            normal_X_train,
            y_train,
            verbose=0,
            epochs=epochs,
            validation_split=0.2
        )
    
    # accuracy
    score = model.evaluate(normal_X_test, y_test)
    acc = score[1]
    
    normal_test = normalizer(test).numpy()
    normal_test = np.nan_to_num(normal_test)
    
    preds = model.predict(normal_test)
    if isClass:
        preds = np.nan_to_num(preds)
        preds = encoder.inverse_transform(preds).flatten()
    preds = preds.flatten()
    
    # predictions
    predictions = []

    for i in range(len(testCopy.index)):
        name = testCopy.iloc[i]['key']
        home_abbr = testCopy.iloc[i]['home_abbr']
        away_abbr = testCopy.iloc[i]['away_abbr']
        predictions.append((name, home_abbr, away_abbr, round(preds[i], 2)))
        
    return predictions, acc

def predictAll(week, year, classFuncs, pa, _type):
    
    df_list, accs = [], []
    string_cols = ['key', 'home_abbr', 'away_abbr']
    alters = [
        (False, True),
        # (False, False),
        # (True, False),
        (True, True)
    ]
    
    for attributes in pa:
        df_list0 = []
        targetType = attributes.targetType
        targetName = attributes.targetName
        if 'won' in targetName or 'touchdown' in targetName or 'interceptions' in targetName:
            isClass = True
            epochs = [500]
            # modelNames = ['log', 'knn']
            modelNames = ['log', 'knn', 'forest']
        else:
            isClass = False
            epochs = [20]
            modelNames = ['log', 'knn', 'knnReg']
        # points epochs
        if targetName == 'points' or 'touchdowns' in targetName:
            epochs = [30]
        print(targetType + "_" + targetName)
        print("isClass:", isClass)
        print("----------------------")
        print("Getting features, labels, and test " + targetType + "_" + targetName + " data.")
        features, labels, test = getFeaturesLabelsTest(targetType, targetName, week, _type)
        labels_i = labels.copy() # copy for individual
        for alt in alters:
            print("Altering data: shorten-" + str(alt[0]) + " & ols-0.1-" + str(alt[1]) + "_" + targetType + "_" + targetName + " data.")
            features, labels, drops = alterData(
                features, 
                labels, 
                shorten=alt[0],
                shortenYear=2005,
                ols=alt[1],
                olsThreshold=0.1,
                saveOls=False,
                fs=False
            )
            # drop test cols if ols/fs is true
            if drops != None:
                test.drop(columns=drops, inplace=True)
            # predict
            for epoch in epochs:
                if targetName != 'points' or 'touchdowns' not in targetName:
                    if isClass:
                        for classFunc in classFuncs:
                            start_time = time.time()
                            preds, acc = predict(features, labels, test.copy(), targetName=targetName, targetType=targetType, epochs=epoch, classFunc=classFunc)
                            pred_name = targetType + "_" + targetName + "_tf_" + classFunc + "_" + str(round(acc, 2)) + "_e" + str(epoch)
                            pred_name += str(alt[0]) + "-" + str(alt[1])
                            print('Epochs: ' + str(epoch) + " -> " + pred_name + " done.")
                            df_list0.append(addPreds(preds, pred_name))
                            accs.append(acc)
                            end_time = time.time()
                            print(pred_name + " -> Time Elapsed: " + str(round(end_time-start_time, 2)))
                    else:
                        start_time = time.time()
                        preds, acc = predict(features, labels, test.copy(), targetName=targetName, targetType=targetType, epochs=epoch, classFunc='')
                        pred_name = targetType + "_" + targetName + "_tf_" + str(round(acc, 2)) + "_e" + str(epoch)
                        pred_name += str(alt[0]) + "-" + str(alt[1])
                        print('Epochs: ' + str(epoch) + " -> " + pred_name + " done.")
                        df_list0.append(addPreds(preds, pred_name))
                        accs.append(acc)
                        end_time = time.time()
                        print(pred_name + " -> Time Elapsed: " + str(round(end_time-start_time, 2)))
                else: # points both class and reg
                    # class
                    for classFunc in classFuncs:
                        start_time = time.time()
                        preds, acc = predict(features, labels, test.copy(), targetName=targetName, targetType=targetType, epochs=epoch, classFunc=classFunc)
                        pred_name = targetType + "_" + targetName + "_class_tf_" + classFunc + "_" + str(round(acc, 2)) + "_e" + str(epoch)
                        pred_name += str(alt[0]) + "-" + str(alt[1])
                        print('Epochs: ' + str(epoch) + " -> " + pred_name + " done.")
                        df_list0.append(addPreds(preds, pred_name))
                        accs.append(acc)
                        end_time = time.time()
                        print(pred_name + " -> Time Elapsed: " + str(round(end_time-start_time, 2)))
                    # reg
                    start_time = time.time()
                    preds, acc = predict(features, labels, test.copy(), targetName=targetName, targetType=targetType, epochs=epoch, classFunc='')
                    pred_name = targetType + "_" + targetName + "_reg_tf_" + str(round(acc, 2)) + "_e" + str(epoch)
                    pred_name += str(alt[0]) + "-" + str(alt[1])
                    print('Epochs: ' + str(epoch) + " -> " + pred_name + " done.")
                    df_list0.append(addPreds(preds, pred_name))
                    accs.append(acc)
                    end_time = time.time()
                    print(pred_name + " -> Time Elapsed: " + str(round(end_time-start_time, 2)))
                # end points
            # sklearn
            for name in modelNames:
                    start_time = time.time()
                    preds, acc = predictSk(features, labels, test.copy(), modelName=name)
                    pred_name = targetType + "_" + targetName + "_" + name + "_" + str(round(acc, 2))
                    pred_name += str(alt[0]) + "-" + str(alt[1])
                    print(pred_name + " done.")
                    df_list0.append(addPreds(preds, pred_name))
                    accs.append(acc)
                    end_time = time.time()
                    print(pred_name + " -> Time Elapsed: " + str(round(end_time-start_time, 2)))
        # add averages
        temp_df = reduce(lambda x, y: pd.merge(x, y, how='left', on=string_cols), df_list0)
        mean_acc = str(round(sum(accs)/len(accs), 2))
        temp_df[targetType + "_" + targetName +'_average_' + mean_acc] = pd.DataFrame({col : pd.to_numeric(temp_df[col]) for col in temp_df.columns if col not in string_cols}).mean(axis=1)
        temp_df[targetType + "_" + targetName +'_average_' + mean_acc] = [round(val, 2) for val in list(temp_df[targetType + "_" + targetName +'_average_' + mean_acc])]
        # most common
        if isClass or targetName == 'points':
            target_cols = [col for col in temp_df.columns if (targetType + "_" + targetName) in col]
            mcList = []
            for index, row in temp_df.iterrows():
                vals = [int(round(row[col], 0)) for col in target_cols]
                m_val = mostCommon(vals)
                mcList.append(m_val)
            temp_df[targetType + "_" + targetName +'_mostCommon'] = mcList
        df_list.append((temp_df, targetType))
        print()
        
    # build week folder
    dir = 'w' + str(week) + "_predictions"
    if dir not in os.listdir("."):
        os.mkdir(dir)
    dir += "/" # save features in week directory
        
    _types = set([a.targetType for a in pa])
    for t in _types:
        temp_list = [d[0] for d in df_list if d[1]==t]
        new_df = reduce(lambda x, y: pd.merge(x, y, how='left', on=string_cols), temp_list)
        new_df.to_csv("%s.csv" % (dir + t + "_" + _type), index=False)
        # keep only average + mostCommon
        keep_cols = [col for col in new_df.columns if 'average' in col or 'mostCommon' in col]
        keep_cols += string_cols
        drop_cols = set(new_df.columns).difference(set(keep_cols))
        new_df.drop(columns=drop_cols, inplace=True)
        new_df.to_csv("%s.csv" % (dir + t + "_s_" + _type), index=False)
    
    return

######################################

start_time = time.time()
    
predictAll(
    week=22,
    year=2022,
    # classFuncs=['softmax', 'sigmoid', 'softplus'],
    classFuncs=['softmax'],
    pa=predict_attributes,
    _type='_ols_e500'
)

end_time = time.time()
print("Time Elapsed: " + str(round(end_time-start_time, 2)))

##################################################

# features, labels, test = getFeaturesLabelsTest('g', 'home_won', 5)

# types = []

# for col in features.columns:
#     if type(features[col].values[0]) is str:
#         print(col)
#         print(features[col].values[:10])
#     types.append(type(features[col].values[0]))
    
# print(set(types))

###########################################

# targetType = 'g'
# targetName = 'home_won'

# features, labels, test = getFeaturesLabelsAndTest(
#     targetType=targetType, 
#     targetName=targetName
# )

# features, labels, drops = alterData(
#     features, 
#     labels, 
#     shorten=True,
#     shortenYear=2005,
#     ols=True,
#     olsThreshold=0.5,
#     saveOls=True,
#     fs=False
# )

# if drops != None:
#     test.drop(columns=drops, inplace=True)

# # gridTuning(features, labels, 'log')

# predict(features,
#         labels,
#         test,
#         'forest',
#         targetName=targetName
# )

#################################

# log, 0.5, shorten:2005
# Accuracy: 0.6956521739130435

# log, 0.5, shorten:none
# Accuracy: 0.6415094339622641

# knn, 0.5, shorten:2005
# Accuracy: 0.5869565217391305

# knn, 0.5, shorten:none
# Accuracy: 0.5660377358490566

# forest, 0.5, shorten:2005
# Accuracy: 0.6086956521739131

# forest, 0.5, shorten:none
# Accuracy: 0.6086956521739131