import pandas as pd
import numpy as np
import os
import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class PredStatsWithWinner:
    def __init__(self, position, _dir):
        self.position = position
        self._dir = _dir
        self.data_dir = self._dir + "data/" + self.position + "/"
        self.features_dir = self._dir + "data/" + self.position + "/features/"
        # frames
        self.cd = pd.read_csv("%s.csv" % ("../data/positionData/" + self.position + "Data"))
        self.cd: pd.DataFrame = self.addDatetimeColumns(self.cd)
        self.df = pd.read_csv("%s.csv" % "../data/gameData")
        self.df: pd.DataFrame = self.addDatetimeColumns(self.df)
        self.target: pd.DataFrame = None
        self.train: pd.DataFrame = None
        # info
        self.source_cols = ['p_id', 'game_key', 'abbr', 'position', 'week', 'year', 'datetime']
        self.target_stats = {
            'QB': ['passing_yards', 'passing_touchdowns']
        }
        self.vol_levels = {
            'QB': 0.5
        }
        self.feature_funcs = [
            self.didWin_feature, self.careerAvg_feature
        ]
        return
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def buildTarget(self):
        cd = self.cd
        cd = cd.loc[(cd['year']>=2000)&(cd['volume_percentage']>=self.vol_levels[self.position])]
        cd = cd[self.source_cols+self.target_stats[self.position]]
        self.saveFrame(cd, (self.data_dir + "target"))
        return
    def didWin_feature(self, source: pd.DataFrame):
        if 'didWin.csv' in os.listdir(self.features_dir):
            print('didWin already exists.')
            return
        df = self.df[['key', 'winning_abbr']]
        df.columns = ['game_key', 'winning_abbr']
        df = source.merge(df, on=['game_key'])
        df['didWin'] = (df['abbr'] == df['winning_abbr']).astype(int)
        df.drop(columns=['winning_abbr'], inplace=True)
        self.saveFrame(df, (self.features_dir + "didWin"))
        return
    def careerAvg_feature(self, source: pd.DataFrame):
        """
        Career average of all target stats for position.
        Args:
            source (pd.DataFrame): source frame
        """
        cd = self.cd
        t_cols = self.target_stats[self.position]
        source = source.tail(20)
        for index, (pid, datetime) in enumerate(source[['p_id', 'datetime']].values):
            stats = cd.loc[
                (cd['p_id']==pid)&
                (cd['datetime']<datetime)&
                (cd['volume_percentage']>=self.vol_levels[self.position]), 
                t_cols
            ].values
            if len(stats) != 0:
                avgs = np.mean(stats, axis=0)
                print(pid, avgs)
        return
    def buildTrain(self):
        self.setTarget()
        source: pd.DataFrame = self.target.drop(columns=self.target_stats[self.position])
        [func(source.copy()) for func in self.feature_funcs]
        for fn in os.listdir(self.features_dir):
            df = pd.read_csv(self.features_dir + fn)
            source = source.merge(df, on=self.source_cols)
        self.saveFrame(source, (self.data_dir + "train"))
        return
    def predict(self):
        self.setTrain()
        self.setTarget()
        data: pd.DataFrame = self.train.merge(self.target, on=self.source_cols)
        for target in self.target_stats[self.position]:
            X = data.drop(columns=self.source_cols+[target])
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"Accuracy: {acc}")
        return
    def setTrain(self):
        self.train = pd.read_csv("%s.csv" % (self.data_dir + "train"))
        return
    def setTarget(self):
        self.target = pd.read_csv("%s.csv" % (self.data_dir + "target"))
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
    
# END / PredStatsWithWinner

###########################

psw = PredStatsWithWinner("QB", "./")

# psw.buildTarget()

psw.buildTrain()

# psw.predict()