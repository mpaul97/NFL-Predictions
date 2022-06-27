import pandas as pd
import numpy as np
import os
import argparse
import locale
import shutil

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

pd.options.mode.chained_assignment = None

TARGET_PATH = "../targets/"

def getFeaturesAndLabels(trainNum, targetName):
    
    features = pd.read_csv("%s.csv" % ("train" + str(trainNum)))
    
    labels = pd.read_csv("%s.csv" % (TARGET_PATH + "target"))
    keep_cols = ['key', 'opp_abbr', 'wy', targetName]
    drop_cols = list(set(labels.columns).difference(set(keep_cols)))
    labels.drop(columns=drop_cols, inplace=True)
    
    return features, labels

def processAttributes(features, X_train, X_test):
    
    cat_cols = ['week', 'season', 'isHome']
    cont_cols = list(set(list(features.columns)).difference(set(cat_cols)))
    cont_cols = list(set(cont_cols).difference(set(['key', 'opp_abbr', 'wy'])))
    
    # standardize continuous features
    cs = MinMaxScaler()
    trainCont = cs.fit_transform(X_train[cont_cols])
    testCont = cs.transform(X_test[cont_cols])
    
    # standardize categorical features
    trainCatList, testCatList = [], []
    for col in cat_cols:
        zb = LabelBinarizer().fit(features[col])
        trainTemp = zb.transform(X_train[col])
        testTemp = zb.transform(X_test[col])
        trainCatList.append(trainTemp)
        testCatList.append(testTemp)
    
    trainCat = np.hstack(trainCatList)
    testCat = np.hstack(testCatList)
    
    X_train = np.hstack([trainCat, trainCont])
    X_test = np.hstack([testCat, testCont])
    
    return (X_train, X_test)

def trainTestSplit(features, labels, testWy, trainTest):
    
    if trainTest:
        
        X_train, X_test, y_train, y_test = train_test_split(features, 
                                                            labels, 
                                                            test_size=0.05, 
                                                            random_state=42
                                                            )
        
    else:
    
        # split by testWy
        X_train = features.loc[features['wy']!=testWy]
        y_train = labels.loc[labels['wy']!=testWy]
        
        X_test = features.loc[features['wy']==testWy]
        y_test = labels.loc[labels['wy']==testWy]
        
    # end if
        
    X_test_copy = X_test.copy()
    
    drops = ['key', 'opp_abbr', 'wy']
    
    # drop keys/strings
    X_train.drop(columns=drops, inplace=True)
    y_train.drop(columns=drops, inplace=True)
    
    X_test.drop(columns=drops, inplace=True)
    y_test.drop(columns=drops, inplace=True)
    
    # standardize points -> [0, 1]
    maxY = labels['points'].max()
    y_train = y_train / maxY
    y_test = y_test / maxY
    
    (X_train, X_test) = processAttributes(features, X_train, X_test)
    
    return X_train, X_test, y_train, y_test, X_test_copy

def getModel(dim, regress):
    
    model = Sequential()
    model.add(Dense(16, input_dim=dim, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(100, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='linear'))
    
    if regress:
        model.add(Dense(1, activation=None))
        
    opt = Adam(learning_rate=1e-3, decay=1e-3/200)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(), 
        optimizer='adam', 
        metrics=['accuracy']
    )
        
    return model

def build_model(hp):
    
    model = Sequential()
    model.add(Dense(hp.Choice('units', [8, 16, 32]), input_dim=180, activation='relu'))
    model.add(Dense(hp.Choice('units', [2, 4, 8]), activation='relu'))
    model.add(Dense(1, activation=None))
    
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(), 
        optimizer='adam', 
        metrics=['accuracy']
    )
    
    return model

########################################

features, labels = getFeaturesAndLabels(14, 'points')

testWy = '20 | 2021'

X_train, X_test, y_train, y_test, X_test_copy = trainTestSplit(features, 
                                                               labels, 
                                                               testWy,
                                                               trainTest=True
                                                               )

model = getModel(X_train.shape[1], regress=True)

model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=8
)

preds = model.predict(X_test)

maxP = labels['points'].max()

for i in range(len(X_test_copy.index)):
    print(X_test_copy.iloc[i]['key'], '-> Pred ->', str(preds[i]*maxP), 'Actual ->', str(y_test.iloc[i]['points']*maxP))
    
test_loss, test_acc = model.evaluate(x=X_test, y=y_test)

print('Accuracy:', test_acc)

######################################
# Tuning

# for fn in os.listdir("."):
#     if fn == 'untitled_project':
#         shutil.rmtree(fn)

# tuner = kt.RandomSearch(
#     build_model,
#     objective='accuracy',
#     max_trials=5
# )

# tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
# best_model = tuner.get_best_models()[0]

# print(best_model)

# models = tuner.get_best_models(num_models=1)
# best_model = models[0]

# best_model.build()
# best_model.summary()
