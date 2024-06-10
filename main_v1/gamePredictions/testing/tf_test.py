import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

pd.options.mode.chained_assignment = None

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
tf.function(experimental_relax_shapes=True)

from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

STR_COLS = ['key', 'wy', 'home_abbr', 'away_abbr']

TARGET_CLASSES = ['home_won']

def getClassModel(normal_X_train, y_train):
    final_dense_val = y_train.shape[1]
    # shape
    input_shape = (normal_X_train.shape[1],)
    # model = Sequential([
    #     layers.Dense((final_dense_val**2), input_shape=input_shape, activation='relu'),
    #     layers.Dropout(0.4),
    #     layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    #     layers.Dense(final_dense_val, activation='sigmoid')
    # ])
    model = Sequential([
        layers.Input(shape=input_shape),
        # layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        # layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(final_dense_val, activation='softplus')
    ])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    tf.function(experimental_relax_shapes=True)
    model.fit(
        normal_X_train,
        y_train,
        verbose=1,
        epochs=50,
        validation_split=0.2,
        batch_size=8,
        callbacks=[early_stopping]
    )
    return model

def getRegModel(normal_X_train, y_train):
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
        epochs=100,
        validation_split=0.2
    )
    return model

def predict(targetName):
    # process data
    train = pd.read_csv("%s.csv" % "../train")
    target = pd.read_csv("%s.csv" % "../target")
    drops_df = pd.read_csv("%s.csv" % "../drops")
    drops = (drops_df[targetName].values[0]).split(",")
    target = target[STR_COLS+[targetName]]
    data = train.merge(target, on=STR_COLS)
    data.drop(columns=STR_COLS+drops, inplace=True)
    
    # split into train test splits
    X = data.drop(columns=[targetName])
    y = data[targetName]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # encode multiclass - onehotencode targets
    encoder = OneHotEncoder()
    encoder.fit((data[targetName].values).reshape(-1, 1))
    y_train = (encoder.transform((y_train.values).reshape(-1, 1))).toarray()
    y_test = (encoder.transform((y_test.values).reshape(-1, 1))).toarray()
    
    # normalize
    normalizer = tf.keras.layers.Normalization(input_shape=[len(X_train.columns), ], axis=-1)
    normalizer.adapt(np.array(X_train))
    normal_X_train = normalizer(X_train).numpy()
    normal_X_test = normalizer(X_test).numpy()
    
    model = getClassModel(normal_X_train, y_train) if targetName in TARGET_CLASSES else getRegModel(normal_X_train, y_train)
    
    # accuracy
    score = model.evaluate(normal_X_test, y_test)
    acc = score[1]
    
    print('Accuracy:', acc)
    
    # predict test
    test = pd.read_csv("%s.csv" % "../test")
    testCopy = test.copy()
    test.drop(columns=STR_COLS+drops, inplace=True)
    
    normal_test = normalizer(test).numpy()
    normal_test = np.nan_to_num(normal_test)
    
    preds = model.predict(normal_test)
    if targetName in TARGET_CLASSES:
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
        
    print(predictions)
        
    return

##############################

targetName = 'home_won'

predict(targetName)

# isPlayoffs, isRival, winsAgainstOpp, losesAgainstOpp, wlpAgainstOpp