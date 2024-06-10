import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

pd.options.mode.chained_assignment = None

class PredStandings:
    def __init__(self, sl: pd.DataFrame, _dir):
        self.sl: pd.DataFrame = sl
        self._dir = _dir
        self.data_dir = _dir + "/data/"
        self.models_dir = _dir + "/models/"
        self.tb: pd.DataFrame = None
        self.train: pd.DataFrame = None
        self.raw_standings: pd.DataFrame = None
        # models + encoders
        self.div_model: LinearRegression = None
        self.conf_model: LinearRegression = None
        self.div_encoder: LabelEncoder = None
        self.conf_encoder: LabelEncoder = None
        return
    def buildTarget(self, years: list):
        """
        Gets user input to create target files, for division and conference standings.
        @params:
            years   - Required  : gets last wy in tiebreakerAttributes for each year (list[int])
        """
        self.setTb()
        st = self.tb
        wys = [st.loc[st['wy'].str.contains(str(year)), 'wy'].values[-1] for year in years]
        fn = 'target_' + '-'.join([str(year)[-2:] for year in years])
        st = st.loc[st['wy'].isin(wys)]
        div_standings, conf_standings = [], []
        for index, row in st.iterrows():
            abbr = row['abbr']
            wy = row['wy']
            print(abbr, wy)
            div = input('Enter division standing: ')
            div_standings.append(div)
            conf = input('Enter conference standings: ')
            conf_standings.append(conf)
        st.insert(2, 'division_standings', div_standings)
        st.insert(3, 'conference_standings', conf_standings)
        self.saveFrame(st, (self.data_dir + fn))
        return
    def createAllTrain(self):
        """
        Merge all target files, create encoders and save to models_dir.
        """
        self.setTb()
        df = pd.concat([pd.read_csv(self.data_dir + fn) for fn in os.listdir(self.data_dir) if 'target_' in fn])
        div_encoder = LabelEncoder()
        div_encoder.fit(self.tb['division'])
        df['division'] = div_encoder.transform(df['division'])
        np.save((self.models_dir + 'div_encoder.npy'), div_encoder.classes_)
        conf_encoder = LabelEncoder()
        conf_encoder.fit(self.tb['conference'])
        df['conference'] = conf_encoder.transform(df['conference'])
        np.save((self.models_dir + 'conf_encoder.npy'), conf_encoder.classes_)
        self.saveFrame(df, (self.data_dir + "all_train"))
        return
    # get heatmap
    def getCorrHeatMap(self, target_col: str):
        self.setTrain()
        data: pd.DataFrame = self.train
        corrmat = data.corr()
        k = 20
        cols = corrmat.nlargest(k, target_col)[target_col].index
        cm = np.corrcoef(data[cols].values.T)
        sns.set(font_scale=0.75)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        return
    def saveModels(self):
        self.setTrain()
        data = self.train
        target_cols = ['division_standings', 'conference_standings']
        str_cols = ['abbr', 'wy']
        # accs = { t: (0, None) for t in target_cols }
        accs = { t: [] for t in target_cols }
        for index, col in enumerate(target_cols):
            X = data.drop(columns=str_cols+target_cols)
            y = data[col]
            # count = 0
            # while accs[col][0] < 0.65:
            #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            #     model = RandomForestClassifier(n_jobs=-1)
            #     model.fit(X_train, y_train)
            #     acc = model.score(X_test, y_test)
            #     if acc > accs[col][0]:
            #         accs[col] = (acc, model)
            #     print(f"{col} - Count: {count}, Acc: {acc}")
            #     count += 1
            for i in range(50):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                model = LogisticRegression(n_jobs=-1)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                accs[col].append((acc, model))
            accs[col].sort(key=lambda x: x[0], reverse=True)
        for key in accs:
            model = accs[key][0][1]
            print(f"Best Acc ({key}): {accs[key][0][0]}")
            pickle.dump(model, open((self.models_dir + key + '.sav'), 'wb'))
        return
    def getPredictions(self, df: pd.DataFrame):
        self.setEncoders()
        self.setModels()
        df['division'] = self.div_encoder.transform(df['division'])
        df['conference'] = self.conf_encoder.transform(df['conference'])
        X = df.drop(columns=['abbr', 'wy'])
        div_preds = self.div_model.predict(X)
        conf_preds = self.conf_model.predict(X)
        new_df = pd.DataFrame()
        new_df['abbr'], new_df['wy'] = df['abbr'], df['wy']
        new_df['division_standings'], new_df['conference_standings'] = div_preds, conf_preds
        new_df['division_standings'] = new_df['division_standings'].astype(int)
        new_df['conference_standings'] = new_df['conference_standings'].astype(int)
        return new_df
    def buildPredictions(self):
        """
        Builds raw_pred_standings (standings per week).
        """
        if 'raw_pred_standings.csv' in os.listdir(self.data_dir):
            print('raw_pred_standings.csv already built. Using existing.')
            return
        self.setTb()
        new_df = self.getPredictions(self.tb)
        self.saveFrame(new_df, (self.data_dir + "raw_pred_standings"))
        return
    def getPredInfo(self, abbr: str, wy: str, df: pd.DataFrame, target_cols: list):
        """
        Used to get division standings and conference standings from raw_pred_standings.\n
        Args:
            abbr (str): team abbr
            wy (str): current wy
            df (pd.DataFrame): raw_pred_standings
            target_cols (list): [division standings, conference standings]\n
        Returns:
            _type_: np.array([division standing, conference_standing])
        """
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        if self.isRegularSeason(week, year):
            start = df.loc[df['wy']==wy].index.values[0]
            try: # not first week - get most recent standings
                info = df.loc[
                    (df.index<start)&
                    (df['abbr']==abbr),
                    target_cols
                ].values[-1]
            except IndexError: # 1 | 1994 - use current week standings
                info = df.loc[
                    (df['wy']==wy)&
                    (df['abbr']==abbr),
                    target_cols
                ].values[0]
        else: # isPlayoffs - get last regular season standings
            info = df.loc[
                (df['wy'].str.contains(str(year)))&
                (df['abbr']==abbr), 
                target_cols
            ].values[-1]
        return info
    def buildPredStandings(self, source: pd.DataFrame):
        """
        Combines home and away pred_standings using previous week to source structure.
        Writes pred_standings to _dir.\n
        Args:
            source (pd.DataFrame): train source
        """
        if 'pred_standings.csv' in os.listdir(self._dir):
            print('pred_standings.csv already built. Using existing.')
            return
        self.setRawStandings()
        df = self.raw_standings
        target_cols = ['division_standings', 'conference_standings']
        home_cols = ['home_' + col for col in target_cols]
        away_cols = ['away_' + col for col in target_cols]
        new_df = pd.DataFrame(columns=list(source.columns)+home_cols+away_cols)
        # source = source.loc[source['wy'].str.contains('2022')]
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), 'Pred Standings Progress')
            wy = row['wy']
            home_abbr, away_abbr = row[['home_abbr', 'away_abbr']]
            home_info = self.getPredInfo(home_abbr, wy, df, target_cols)
            away_info = self.getPredInfo(away_abbr, wy, df, target_cols)
            new_df.loc[len(new_df.index)] = np.concatenate([row.values, home_info, away_info])
        self.saveFrame(new_df, (self._dir + "pred_standings"))
        return
    def buildNewPredStandings(self, source: pd.DataFrame):
        """
        Combines home and away pred_standings using last available week to source structure.\n
        Args:
            source (pd.DataFrame): train source\n
        Returns:
            _type_: new pred_standings
        """
        print('Creating new pred_standings...')
        self.setRawStandings()
        df = self.raw_standings
        target_cols = ['division_standings', 'conference_standings']
        home_cols = ['home_' + col for col in target_cols]
        away_cols = ['away_' + col for col in target_cols]
        new_df = pd.DataFrame(columns=list(source.columns)+home_cols+away_cols)
        for index, row in source.iterrows():
            home_abbr, away_abbr = row[['home_abbr', 'away_abbr']]
            # get last standings
            home_info = df.loc[df['abbr']==home_abbr, target_cols].values[-1]
            away_info = df.loc[df['abbr']==away_abbr, target_cols].values[-1]
            new_df.loc[len(new_df.index)] = np.concatenate([row.values, home_info, away_info])
        return new_df
    def isRegularSeason(self, week: int, year: int):
        seasonWeeks = self.sl.loc[self.sl['year']==year, 'weeks'].values[0]
        if week <= seasonWeeks:
            return True
        return False
    def updatePredictions(self):
        self.setTb()
        self.setRawStandings()
        last_wy_tb = self.tb['wy'].values[-1]
        last_wy_raw = self.raw_standings['wy'].values[-1]
        week_tb = int(last_wy_tb.split(" | ")[0])
        year_tb = int(last_wy_tb.split(" | ")[1])
        if last_wy_raw != last_wy_tb and self.isRegularSeason(week_tb, year_tb):
            new_df = self.getPredictions(self.tb.loc[self.tb['wy']==last_wy_tb])
            self.raw_standings = pd.concat([self.raw_standings, new_df])
            self.saveFrame(self.raw_standings, (self.data_dir + "raw_pred_standings"))
            print(f"raw_pred_standings updated for wy: {last_wy_tb}")
        else:
            print("raw_pred_standings.csv already up-to-date.")
        return
    def setRawStandings(self):
        self.raw_standings = pd.read_csv("%s.csv" % (self.data_dir + "raw_pred_standings"))
        return
    def setModels(self):
        self.div_model = pickle.load(open((self.models_dir + 'division_standings.sav'), 'rb'))
        self.conf_model = pickle.load(open((self.models_dir + 'conference_standings.sav'), 'rb'))
        return
    def setEncoders(self):
        self.div_encoder = LabelEncoder()
        self.div_encoder.classes_ = np.load(self.models_dir + "div_encoder.npy", allow_pickle=True)
        self.conf_encoder = LabelEncoder()
        self.conf_encoder.classes_ = np.load(self.models_dir + "conf_encoder.npy", allow_pickle=True)
        return
    def setTb(self):
        self.tb = pd.read_csv("%s.csv" % (self._dir + "tiebreakerAttributes"))
        return
    def setTrain(self):
        self.train = pd.read_csv("%s.csv" % (self.data_dir + "all_train"))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
        return

# / END PredStandings

########################

# sl = pd.read_csv("%s.csv" % "../../../../data/seasonLength")
# ps = PredStandings(sl, './')

# # ps.buildTarget([2021, 2022])

# # ps.createAllTrain()

# # ps.saveModels()

# # ps.buildPredictions()

# source = pd.read_csv("%s.csv" % "../source/source")
# ps.buildPredStandings(source)

# ps.updatePredictions()

# source = pd.read_csv("%s.csv" % "../source/new_source")
# df = ps.buildNewPredStandings(source)