import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import NuSVC, NuSVR, LinearSVC, LinearSVR

import sys
sys.path.append("../../")

from paths import ALL_PATHS

try:
    from features.source import Source
    from main.games.features.team_elos import TeamElos
    from main.games.features.season_rankings import SeasonRankings
except ModuleNotFoundError:
    from games.features.source import Source
    from main.games.features.team_elos import TeamElos
    from main.games.features.season_rankings import SeasonRankings

class Build:
    def __init__(self, paths: str = None):
        self.paths = paths if paths else { k: ALL_PATHS[k].replace("games/", "") for k in ALL_PATHS }
        self._dir: str = os.getcwd()[:os.getcwd().index("main")+len("main")]+"/"
        self.data_dir = self._dir + "games/data/"
        self.models_dir = self._dir + "games/models/"
        self.features_dir = self._dir + "games/features/data/"
        # dataframes
        self.df: pd.DataFrame = pd.read_csv("%s.csv" % (self.paths['DATA_PATH'] + "gameData"))
        self.source: pd.DataFrame = None
        self.source_new: pd.DataFrame = None
        self.target: pd.DataFrame = None
        # other
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        self.target_cols = ['home_won', 'home_points', 'away_points']
        # feature objects
        self.SOURCE = Source()
        self.TEAM_ELOS = TeamElos(self.df, self.features_dir)
        self.SEASON_RANKINGS = SeasonRankings(self.df, self.features_dir)
        # feature names
        self.feature_names = [
            'team_elos', 'season_rankings'
        ]
        return
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def set_source(self):
        self.source = pd.read_csv("%s.csv" % (self.features_dir + "source"))
        return
    def set_source_new(self):
        self.source_new = pd.read_csv("%s.csv" % (self.features_dir + "source_new"))
        return
    def build_targets(self):
        """
        Create targets
        """
        if 'targets.csv' in os.listdir(self.data_dir):
            print('targets already exists.')
            self.target = pd.read_csv("%s.csv" % (self.data_dir + "targets"))
            return
        df: pd.DataFrame = self.source.copy().merge(self.df[['key', 'winning_abbr', 'home_points', 'away_points']], on=['key'])
        df['home_won'] = df.apply(lambda row: 1 if row['winning_abbr'] == row['home_abbr'] else 0, axis=1)
        df.drop(columns=['winning_abbr'], inplace=True)
        self.save_frame(df, (self.data_dir + "targets"))
        self.target = df
        return
    def train(self):
        """
        Train/save all models
        """
        df: pd.DataFrame = self.source.copy()
        for fn in self.feature_names:
            cd: pd.DataFrame = pd.read_csv("%s.csv" % (self.features_dir + fn))
            df = df.merge(cd, on=self.str_cols)
        self.save_frame(df, (self.data_dir + "train"))
        df = df.merge(self.target, on=self.str_cols)
        # train
        X: pd.DataFrame = df.drop(columns=self.str_cols+self.target_cols)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pickle.dump(scaler, open((self.models_dir + 'scaler.sav'), 'wb'))
        for t_col in self.target_cols:
            y = df[t_col]
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
            model = LogisticRegression(n_jobs=-1, max_iter=2000)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"{t_col} - accuracy: {acc}")
            pickle.dump(model, open((self.models_dir + t_col + '.sav'), 'wb'))
        print("Models saved.")
        print()
        return
    def run(self):
        """
        Build all features, merge, targets, and train
        """
        # source
        self.SOURCE.build()
        self.set_source()
        # features
        # ----------
        print()
        self.TEAM_ELOS.build()
        self.TEAM_ELOS.createBoth(self.source.copy())
        #
        self.SEASON_RANKINGS.build(self.source.copy())
        # ----------
        print()
        self.build_targets()
        self.train()
        return
    def predict(self):
        """
        Load models and create predictions
        """
        df: pd.DataFrame = self.source_new.copy()
        pdf: pd.DataFrame = self.source_new.copy()
        for fn in self.feature_names:
            cd: pd.DataFrame = pd.read_csv("%s.csv" % (self.features_dir + fn + "_new"))
            df = df.merge(cd, on=self.str_cols)
        self.save_frame(df, (self.data_dir + "test"))
        # predict
        scaler: StandardScaler = pickle.load(open((self.models_dir + "scaler.sav"), "rb"))
        X: pd.DataFrame = df.drop(columns=self.str_cols)
        X_scaled = scaler.transform(X)
        for t_col in self.target_cols:
            model: LogisticRegression = pickle.load(open((self.models_dir + t_col + '.sav'), 'rb'))
            preds = model.predict(X_scaled)
            pdf[t_col] = preds
        self.save_frame(pdf, (self.data_dir + "predictions"))
        print("Predictions saved.")
        print()
        return
    def update(self):
        """
        Update all features
        """
        self.TEAM_ELOS.update()
        return
    def run_new(self, week: int, year: int):
        """
        Build all features, merge, and predict
        """
        print('Updating data...')
        self.update()
        print()
        # source
        self.SOURCE.build_new(week, year)
        self.set_source_new()
        # features
        # ----------
        print()
        self.TEAM_ELOS.createBoth(self.source_new.copy(), isNew=True)
        #
        self.SEASON_RANKINGS.build(self.source_new.copy(), isNew=True)
        # ----------
        print()
        self.predict()
        return

##############################
    
if __name__ == '__main__':
    Build().run()
    # Build().run_new(1, 2024)
