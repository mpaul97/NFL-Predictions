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

class Main:
    def __init__(self, _dir):
        self._dir = _dir
        self.train = pd.read_csv("%s.csv" % (self._dir + "../train"))
        self.pred_train = pd.read_csv("%s.csv" % (self._dir + "../pred_train"))
        self.target = pd.read_csv("%s.csv" % (self._dir + "../target"))
        self.test = pd.read_csv("%s.csv" % (self._dir + "../test"))
        self.pred_test = pd.read_csv("%s.csv" % (self._dir + "../pred_test"))
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        self.target_cols = ['home_won', 'home_points', 'away_points']
        self.class_targets = ['home_won']
        self.target_labels = {
            'home_won': [0, 1]
        }
        return
    def getModel_class(self, X_train_scaled: pd.DataFrame, y_train: pd.DataFrame):
        model = Sequential([
            layers.Input(shape=(X_train_scaled.shape[1],)),
            # layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            # layers.Dropout(0.4),
            layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(2, activation='softplus')
        ])
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        # Train the model
        model.fit(
            X_train_scaled, 
            y_train, 
            epochs=50, 
            batch_size=8, 
            verbose=1,
            callbacks=[early_stopping]
        )
        return model
    def predsToLabels(self, target_name: str, preds: np.array):
        labels = []
        for i in range(preds.shape[0]):
            arr = list(preds[i])
            max_idx = arr.index(max(arr))
            labels.append(self.target_labels[target_name][max_idx])
        return labels
    def predict(self, pred: bool):
        train: pd.DataFrame = self.pred_train if pred else self.train
        for col in self.target_cols:
            X = train.drop(columns=self.str_cols)
            y = self.target[col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            if col in self.class_targets:
                model: Sequential = self.getModel_class(X_train_scaled, y_train)
                loss, accuracy = model.evaluate(X_test_scaled, y_test)
                print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
                # predict new test
                self.test = self.pred_test if pred else self.test
                test = self.test.drop(columns=self.str_cols)
                test_scaled = scaler.transform(test)
                preds = model.predict(test_scaled)
                pred_labels = self.predsToLabels(col, preds)
                self.test[col] = pred_labels
                self.test = self.test[self.str_cols + [col]]
                fn = "tf_pred_predictions" if pred else "tf_predictions" 
                # self.saveFrame(self.test, (self._dir + "../" + fn))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return

# END / Tf

####################

m = Main("./")

m.predict(pred=False)
# m.predict(pred=True)