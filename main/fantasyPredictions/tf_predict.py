import pandas as pd
import numpy as np
import os
from functools import reduce

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

class TfPredict:
    def __init__(self, _dir):
        self._dir = _dir
        self.positions = ['QB', 'RB', 'WR', 'TE']
        self.str_cols = ['key', 'abbr', 'p_id', 'wy', 'position']
        # frames
        self.train = pd.read_csv("%s.csv" % (self._dir + "train"))
        self.test = pd.read_csv("%s.csv" % (self._dir + "test"))
        self.target = pd.read_csv("%s.csv" % (self._dir + "target"))
        self.pred_train = pd.read_csv("%s.csv" % (self._dir + "pred_train"))
        self.pred_test = pd.read_csv("%s.csv" % (self._dir + "pred_test"))
        return
    def getModel_reg(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int):
        input_shape = (X_train.shape[1],)
        model = Sequential([
            layers.Dense(64, input_shape=input_shape, activation='relu'),
            layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dense(1, activation='linear')
        ])
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        # model.compile(
        #     loss=tf.keras.losses.mae,
        #     optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
        #     metrics=['accuracy']
        # )
        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['mean_absolute_error']
        )
        tf.function(experimental_relax_shapes=True)
        model.fit(
            X_train,
            y_train,
            verbose=0,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        return model
    def predict(self, use_pred: bool):
        train = self.pred_train if use_pred else self.train
        test = self.pred_test if use_pred else self.test
        pred_col = 'pred_tf_points' if use_pred else 'tf_points'
        df_list = []
        for position in self.positions:
            drop_cols = [col for col in train.columns if 'pred' in col and position not in col] if use_pred else []
            X = train.loc[train['position']==position].drop(columns=self.str_cols+drop_cols)
            y = self.target.loc[self.target['position']==position, 'points']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # normalize
            normalizer = layers.Normalization(input_shape=[len(X_train.columns), ], axis=-1)
            normalizer.adapt(np.array(X_train))
            normal_X_train = normalizer(X_train).numpy()
            normal_X_test = normalizer(X_test).numpy()
            # get model
            model: Sequential = self.getModel_reg(normal_X_train, y_train, epochs=50)
            score = model.evaluate(normal_X_test, y_test)
            metric = 'mse'
            print(f"{position} - {metric}: {score[1]}")
            # make predictions
            new_df = test.loc[test['position']==position]
            n_test = new_df.drop(columns=self.str_cols+drop_cols)
            n_test = normalizer(n_test).numpy()
            n_test = np.nan_to_num(n_test)
            preds = model.predict(n_test)
            preds = preds.flatten()
            new_df[pred_col] = preds
            df_list.append(new_df)
        all_df = pd.concat(df_list)
        all_df = all_df[self.str_cols+[pred_col]]
        all_df = all_df.sort_values(by=[pred_col], ascending=False)
        all_df.to_csv("%s.csv" % (self._dir + "tf_predictions"), index=False)
        return
    def build(self):
        self.predict(False)
        self.predict(True)
        return
    
# END / TfPredict

#######################

# tfp = TfPredict("./")

# tfp.build()