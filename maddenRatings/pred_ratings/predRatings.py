import pandas as pd
import numpy as np
import os
import datetime

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor

class PredRatings:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.test_dir = self._dir + "test/"
        self.position_dir = self._dir + "../../data/positionData/"
        # oline, kickers, punters, unknown positions => use mean
        self.positions = ['QB', 'RB', 'WR', 'TE', 'LB', 'DL', 'DB']
        self.frame_positions = ['QB', 'RB', 'WR', 'TE', 'LBDL', 'DB']
        self.non_stat_cols = ['abbr', 'game_key', 'isHome', 'position']
        self.stat_frames: dict = {}
        return
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def initStatFrames(self):
        frames = {}
        for pos in self.frame_positions:
            df = pd.read_csv("%s.csv" % (self.position_dir + pos + "Data"))
            df = df.drop(columns=self.non_stat_cols)
            df = self.addDatetimeColumns(df)
            frames[pos] = df
        return frames
    def buildData(self):
        """
        Using career averages of all stats for each position
        """
        self.stat_frames = self.initStatFrames()
        df = pd.read_csv("%s.csv" % (self._dir + "../allOverallRatings_01-23"))
        df = df.loc[df['p_id']!='UNK'] # no UNK pids
        df = df.loc[df['position'].isin(self.positions)] # no UNK_POS, OL, LS, etc.
        # df = df.loc[(df['position']=='QB')&(df['year']==2023)]
        train_frames = { pos: [] for pos in self.frame_positions }
        for index, (pid, position, year, rating) in enumerate(df[['p_id', 'position', 'year', 'overall_rating']].values):
            self.printProgressBar(index, len(df.index), 'Data')
            f_position = position if position not in ['LB', 'DL'] else 'LBDL'
            cd: pd.DataFrame = self.stat_frames[f_position]
            stats: pd.DataFrame = cd.loc[(cd['p_id']==pid)&(cd['year']<year)]
            if not stats.empty:
                stats = stats.drop(columns=['p_id', 'wy', 'week', 'year', 'datetime'])
                avgs = stats.sum().to_frame().transpose()
                avgs.insert(0, 'p_id', pid)
                avgs.insert(1, 'year', year)
                avgs['rating'] = rating
                train_frames[f_position].append(avgs)
            else:
                avgs = pd.DataFrame(columns=cd.columns)
                avgs.loc[len(avgs.index)] = np.zeros(len(cd.columns))
                avgs = avgs.drop(columns=['wy', 'week', 'year', 'datetime'])
                avgs['p_id'] = pid
                avgs.insert(1, 'year', year)
                avgs['rating'] = rating
                train_frames[f_position].append(avgs)
        for pos in self.frame_positions:
            try:
                new_df = pd.concat(train_frames[pos])
                self.saveFrame(new_df, (self.data_dir + pos + "Data"))
            except ValueError:
                print(f"Position {pos} empty.")
        return
    def buildTest(self):
        self.stat_frames = self.initStatFrames()
        years = [i for i in range(1994, 2001)]
        for pos in self.frame_positions:
            df_list = []
            cd: pd.DataFrame = self.stat_frames[pos]
            for year in years:
                print(pos, year)
                pids = list(set(cd.loc[cd['year']==year, 'p_id'].values))
                for pid in pids:
                    stats: pd.DataFrame = cd.loc[(cd['p_id']==pid)&(cd['year']<year)]
                    if not stats.empty:
                        stats = stats.drop(columns=['p_id', 'wy', 'week', 'year', 'datetime'])
                        avgs = stats.sum().to_frame().transpose()
                        avgs.insert(0, 'p_id', pid)
                        avgs.insert(1, 'year', year)
                        df_list.append(avgs)
                    else:
                        avgs = pd.DataFrame(columns=cd.columns)
                        avgs.loc[len(avgs.index)] = np.zeros(len(cd.columns))
                        avgs = avgs.drop(columns=['wy', 'week', 'year', 'datetime'])
                        avgs['p_id'] = pid
                        avgs.insert(1, 'year', year)
                        df_list.append(avgs)
            new_df = pd.concat(df_list)
            self.saveFrame(new_df, (self.test_dir + pos + "Test"))
        return
    def getDrops(self, X, y, threshold):
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        pdf: pd.Series = results.pvalues.sort_values()
        ols = pdf.to_frame()
        ols.insert(0, 'name', ols.index)
        ols.columns = ['name', 'val']
        ols.fillna(1, inplace=True)
        drops = []
        for index, row in ols.iterrows():
            name = row['name']
            val = row['val']
            if val > threshold and name != 'const':
                drops.append(name)
        return drops
    def predict(self):
        df_list = []
        for pos in self.frame_positions:
            data = pd.read_csv("%s.csv" % (self.data_dir + pos + "Data"))
            test = pd.read_csv("%s.csv" % (self.test_dir + pos + "Test"))
            X = data.drop(columns=['p_id', 'year', 'rating'])
            y = data['rating']
            drops = self.getDrops(X, y, 0.5)
            X = X.drop(columns=drops)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestRegressor(n_jobs=-1)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"Position: {pos}, Accuracy: {acc}")
            n_test = test.drop(columns=['p_id', 'year']+drops)
            preds = model.predict(n_test)
            test = test[['p_id', 'year']]
            test.insert(1, 'position', pos)
            test['overall_rating'] = [round(p, 0) for p in preds]
            test.sort_values(by=['year'], inplace=True)
            df_list.append(test)
        self.saveFrame(pd.concat(df_list), 'predOverallRatings_94-00')
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def printProgressBar(self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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
    
# END / PredRatings

#########################

pr = PredRatings("./")

# pr.buildData()

# pr.buildTest()

pr.predict()