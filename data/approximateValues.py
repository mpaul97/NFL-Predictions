import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None

class ApproximateValues:
    def __init__(self, _dir):
        self._dir = _dir
        self.models_dir = self._dir + "models/"
        self.cd = pd.read_csv("%s.csv" % (self._dir + "advancedStats"))
        self.str_cols = ['player_id', 'season', 'team_abbreviation', 'position']
        return
    def convertRecord(self, record: str):
        if type(record) == float:
            return 0
        wins, loses, ties = record.split("-")
        wins, loses, ties = int(wins), int(loses), int(ties)
        return (2*(wins+ties))/(2*sum([wins, loses, ties]))
    def save_model(self):
        cd = self.cd.loc[
            (~pd.isna(self.cd['approximate_value']))&
            (self.cd['position']=='QB')
        ]
        cd = cd.reset_index(drop=True)
        cd['qb_record'] = cd['qb_record'].apply(lambda x: self.convertRecord(x))
        cd.fillna(0, inplace=True)
        X = cd.drop(columns=self.str_cols+['approximate_value'])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pickle.dump(scaler, open((self.models_dir + "approximate_values_scaler.sav"), "wb"))
        y = cd['approximate_value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression(n_jobs=-1)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"Accuracy: {acc}")
        pickle.dump(model, open((self.models_dir + "approximate_values.sav"), "wb"))
        return
    def predict(self):
        """
        Use saved model to predict 2023 missing approximate values
        """
        print("Predicting 2023 approximate values...")
        cd = self.cd.loc[
            (self.cd['season']=='2023')&
            (self.cd['position']=='QB')
        ]
        cd['qb_record'] = cd['qb_record'].apply(lambda x: self.convertRecord(x))
        cd.fillna(0, inplace=True)
        scaler: StandardScaler = pickle.load(open((self.models_dir + "approximate_values_scaler.sav"), "rb"))
        X = cd.drop(columns=self.str_cols+['approximate_value'])
        X = scaler.transform(X)
        model: LinearRegression = pickle.load(open((self.models_dir + "approximate_values.sav"), "rb"))
        preds = model.predict(X)
        for i, index in enumerate(cd.index):
            p = preds[i]
            self.cd.at[index, 'approximate_value'] = round(p, 1)
        self.cd.to_csv("%s.csv" % (self._dir + "advancedStats"), index=False)
        return
    
# END / ApproximateValues

#############################

# av = ApproximateValues("./")

# # av.save_model()
# av.predict()