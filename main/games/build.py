import pandas as pd
import numpy as np
import os
import pickle
import statsmodels.api as sm

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
    from main.games.features.vegas_lines import VegasLines
    from main.games.features.advanced_starter_stats import AdvancedStarterStats
    from main.games.features.last_avg_n import LastAvgN
except ModuleNotFoundError:
    from games.features.source import Source
    from games.features.team_elos import TeamElos
    from games.features.season_rankings import SeasonRankings
    from games.features.vegas_lines import VegasLines
    from games.features.advanced_starter_stats import AdvancedStarterStats
    from games.features.last_avg_n import LastAvgN

class Build:
    def __init__(self, paths: str = None):
        self.paths = paths if paths else { k: ALL_PATHS[k].replace("games/", "") for k in ALL_PATHS }
        self._dir: str = os.getcwd()[:os.getcwd().index("main")+len("main")]+"/"
        self.data_dir = self._dir + "games/data/"
        self.models_dir = self._dir + "games/models/"
        self.other_dir = self._dir + "games/other/"
        self.features_dir = self._dir + "games/features/data/"
        # dataframes
        self.df: pd.DataFrame = pd.read_csv("%s.csv" % (self.paths['DATA_PATH'] + "gameData"))
        self.old_df: pd.DataFrame = pd.read_csv("%s.csv" % (self.paths['DATA_PATH'] + "oldGameData_94"))
        self.adf: pd.DataFrame = pd.read_csv("%s.csv" % (self.paths['DATA_PATH'] + "advancedStats"))
        self.tn: pd.DataFrame = pd.read_csv("%s.csv" % (self.paths['TEAMNAMES_PATH'] + "teamNames_line"))
        self.sdf: pd.DataFrame = pd.read_csv("%s.csv" % (self.paths['STARTERS_PATH'] + "allStarters"))
        self.sdf_new: pd.DataFrame = None
        self.source: pd.DataFrame = None
        self.source_new: pd.DataFrame = None
        self.target: pd.DataFrame = None
        self.drops: pd.DataFrame = None
        # other
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        self.target_cols = ['home_won', 'home_points', 'away_points']
        # feature objects
        self.SOURCE = Source()
        self.TEAM_ELOS = TeamElos(self.df, self.features_dir)
        self.SEASON_RANKINGS = SeasonRankings(self.df, self.features_dir)
        self.VEGAS_LINES = VegasLines(self.df, self.tn, self.features_dir)
        self.QB_ADVANCED_STARTER_STATS = AdvancedStarterStats('qb', self.sdf, self.adf, self.features_dir)
        self.LAST_AVG_N_5 = LastAvgN(5, self.old_df, self.features_dir)
        # feature names
        self.feature_names = [
            'team_elos', 'season_rankings', 'vegaslines',
            'qb_advanced_starter_stats', 'last_avg_n_5'
        ]
        # model configs
        self.thresholds = {
            'home_won': 0.5, 'home_points': 0.5, 'away_points': 0.5
        }
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
    def set_drops(self):
        self.drops = pd.read_csv("%s.csv" % (self.data_dir + "drops"))
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
    def ols(self, X: np.ndarray, y: pd.Series, target_name: str):
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        pdf: pd.Series = results.pvalues.sort_values()
        ols = pdf.to_frame()
        ols.insert(0, 'name', ols.index)
        ols.columns = ['name', 'val']
        ols.fillna(1, inplace=True)
        self.save_frame(ols, (self.other_dir + target_name + "_ols"))
        return list(ols.loc[(ols['val']>self.thresholds[target_name])&(ols['name']!='const'), 'name'].values)
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
        ddf: pd.DataFrame = pd.DataFrame()
        for t_col in self.target_cols:
            y = df[t_col]
            drops: list[str] = self.ols(X, y, t_col)
            ddf[t_col] = ["|".join(drops)]
            X1: pd.DataFrame = X.copy().drop(columns=drops)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X1)
            pickle.dump(scaler, open((self.models_dir + t_col + '_scaler.sav'), 'wb'))
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
            model = LogisticRegression(n_jobs=-1, max_iter=2000)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"{t_col} - accuracy: {acc}")
            pickle.dump(model, open((self.models_dir + t_col + '.sav'), 'wb'))
        print("Models saved.")
        self.save_frame(ddf, (self.data_dir + "drops"))
        print("Drops saved.")
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
        #
        self.VEGAS_LINES.build(self.source.copy())
        #
        self.QB_ADVANCED_STARTER_STATS.build(self.source.copy())
        #
        self.LAST_AVG_N_5.build(self.source.copy())
        # ----------
        print()
        self.build_targets()
        self.train()
        return
    def predict(self):
        """
        Load models and create predictions
        """
        wy: str = self.source_new['wy'].values[0]
        df: pd.DataFrame = self.source_new.copy()
        pdf: pd.DataFrame = self.source_new.copy()
        for fn in self.feature_names:
            fn = (fn + "_new")
            fn = fn if (fn + ".csv") in os.listdir(self.features_dir) else (fn + "_" + wy.replace(" | ","-"))
            cd: pd.DataFrame = pd.read_csv("%s.csv" % (self.features_dir + fn))
            df = df.merge(cd, on=self.str_cols)
        self.save_frame(df, (self.data_dir + "test"))
        # predict
        self.set_drops()
        X: pd.DataFrame = df.drop(columns=self.str_cols)
        for t_col in self.target_cols:
            scaler: StandardScaler = pickle.load(open((self.models_dir + t_col + "_scaler.sav"), "rb"))
            X1: pd.DataFrame = X.copy().drop(columns=(self.drops[t_col].values[0]).split("|"))
            X_scaled = scaler.transform(X1)
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
        # curr wy
        wy = str(week) + " | " + str(year)
        # source
        self.SOURCE.build_new(week, year)
        self.set_source_new()
        # starters
        if wy in self.sdf['wy'].values:
            self.sdf_new = self.sdf.loc[self.sdf['wy']==wy]
        else:
            s_dir = 'starters_' + str(year)[-2:] + '/'
            self.sdf_new = pd.read_csv("%s.csv" % (self.paths['DATA_PATH'] + s_dir + "starters_w" + str(week)))
        # features
        # ----------
        print()
        self.TEAM_ELOS.createBoth(self.source_new.copy(), isNew=True)
        #
        self.SEASON_RANKINGS.build(self.source_new.copy(), isNew=True)
        #
        self.VEGAS_LINES.build(self.source_new.copy(), isNew=True)
        #
        self.QB_ADVANCED_STARTER_STATS.sdf = self.sdf_new
        self.QB_ADVANCED_STARTER_STATS.build(self.source_new.copy(), isNew=True)
        #
        self.LAST_AVG_N_5.build(self.source_new.copy(), isNew=True)
        # ----------
        print()
        self.predict()
        return

##############################
    
if __name__ == '__main__':
    # Build().run()
    Build().run_new(1, 2024)
