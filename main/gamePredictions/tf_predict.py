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
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        self.all_targets = ['home_won', 'home_points', 'away_points']
        self.class_targets = ['home_won']
        # frames
        self.train = pd.read_csv("%s.csv" % (self._dir + "train"))
        self.test = pd.read_csv("%s.csv" % (self._dir + "test"))
        self.target = pd.read_csv("%s.csv" % (self._dir + "target"))
        self.pred_train = pd.read_csv("%s.csv" % (self._dir + "pred_train"))
        self.pred_test = pd.read_csv("%s.csv" % (self._dir + "pred_test"))
        # all models
        self.all_models = {
            'regression': self.getModel_reg, 'class': self.getModel_class
        }
        return
    def most_common(self, lst):
        return max(set(lst), key=lst.count)
    def getModel_reg(self, X_train: np.ndarray, y_train: np.ndarray):
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
            verbose=1,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        return model
    def getModel_class(self, X_train: np.ndarray, y_train: np.ndarray):
        final_dense_val = y_train.shape[1]
        # shape
        input_shape = (X_train.shape[1],)
        model = Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(final_dense_val, activation='sigmoid')
        ])
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        tf.function(experimental_relax_shapes=True)
        model.fit(
            X_train,
            y_train,
            verbose=1,
            epochs=50,
            validation_split=0.2,
            batch_size=8,
            callbacks=[early_stopping]
        )
        return model
    def predict(self, use_pred: bool):
        train = self.pred_train if use_pred else self.train
        test = self.pred_test if use_pred else self.test
        test_copy = test.copy()
        X = train.drop(columns=self.str_cols)
        for t_name in self.all_targets:
            y = self.target[t_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            if t_name in self.class_targets: # encode multiclass - onehotencode targets
                encoder = OneHotEncoder()
                encoder.fit((y.values).reshape(-1, 1))
                y_train = (encoder.transform((y_train.values).reshape(-1, 1))).toarray()
                y_test = (encoder.transform((y_test.values).reshape(-1, 1))).toarray()
            # normalize
            normalizer = layers.Normalization(input_shape=[len(X_train.columns), ], axis=-1)
            normalizer.adapt(np.array(X_train))
            normal_X_train = normalizer(X_train).numpy()
            normal_X_test = normalizer(X_test).numpy()
            # get model
            model_name = 'class' if t_name in self.class_targets else 'regression'
            model: Sequential = self.all_models[model_name](normal_X_train, y_train)
            score = model.evaluate(normal_X_test, y_test)
            metric = 'accuracy' if t_name in self.class_targets else 'mse'
            print(f"{t_name} - {metric}: {score[1]}")
            # make predictions
            n_test = test_copy.drop(columns=self.str_cols)
            n_test = normalizer(n_test).numpy()
            n_test = np.nan_to_num(n_test)
            preds = model.predict(n_test)
            if t_name in self.class_targets:
                preds = np.nan_to_num(preds)
                preds = encoder.inverse_transform(preds).flatten()
            preds = preds.flatten()
            test[t_name] = preds
        test = test[self.str_cols+self.all_targets]
        test = test.round(0)
        point_cols = [col for col in test.columns if 'points' in col]
        point_cols = list(set(['_'.join(col.split("_")[1:]) for col in point_cols]))
        for col in point_cols:
            test[col + '_h_won'] = test.apply(lambda x: 1 if x['home_' + col] > x['away_' + col] else 0, axis=1)
        fn = "tf_pred_predictions" if use_pred else "tf_predictions"
        self.saveFrame(test, (self._dir + fn))
        return
    def build(self):
        self.predict(use_pred=False)
        self.predict(use_pred=True)
        return
    def buildConsensus(self):
        fns = [fn for fn in os.listdir(self._dir) if 'predictions' in fn]
        df_list, won_cols = [], []
        for fn in fns:
            df = pd.read_csv(self._dir + fn)
            cols = [col for col in df.columns if 'won' in col]
            df = df[self.str_cols+cols]
            df.columns = self.str_cols + [(fn.replace('.csv','') + '_' + col) for col in cols]
            [won_cols.append(col) for col in df.columns if col not in self.str_cols]
            df_list.append(df)
        new_df: pd.DataFrame = reduce(lambda x, y: pd.merge(x, y, on=self.str_cols), df_list)
        new_df['most_common'] = new_df.apply(lambda x: self.most_common(list(x[won_cols])), axis=1)
        new_df['consensus'] = new_df.apply(lambda x: round(((list(x[won_cols]).count(x['most_common'])) / len(won_cols))*100, 2), axis=1)
        new_df['consensus_abbr'] = new_df.apply(lambda x: x['home_abbr'] if x['most_common']==1 else x['away_abbr'], axis=1)
        self.saveFrame(new_df, (self._dir + 'consensus'))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    
    
# END / TfPredict

###########################

# tfp = TfPredict("./")

# # tfp.build()

# tfp.buildConsensus()