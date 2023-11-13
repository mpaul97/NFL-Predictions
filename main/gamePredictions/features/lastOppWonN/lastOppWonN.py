import pandas as pd
import numpy as np
import os
import datetime

class LastOppWonN:
    def __init__(self, df: pd.DataFrame, _dir):
        self.df = df
        self.df: pd.DataFrame = self.addDatetimeColumns(self.df)
        self._dir = _dir
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def getStats(self, n: int, abbr: str, opp_abbr: str, dt: datetime):
        self.df = self.df[['datetime', 'home_abbr', 'away_abbr', 'winning_abbr']]
        try:
            stats: pd.DataFrame = self.df.loc[
                (((self.df['home_abbr']==abbr)&(self.df['away_abbr']==opp_abbr))|
                ((self.df['away_abbr']==abbr)&(self.df['home_abbr']==opp_abbr)))&
                (self.df['datetime']<dt),
            ].tail(n)
            stats['abbr_won'] = stats.apply(lambda x: 1 if abbr==x['winning_abbr'] else 0, axis=1)
            stats = stats['abbr_won'].values[::-1]
        except ValueError:
            stats = np.zeros(n)
        if len(stats) < n:
            dif = n - len(stats)
            stats = np.concatenate((stats, np.zeros(dif)))
        return list(stats)
    def buildLastOppWonN(self, n: int, source: pd.DataFrame, isNew: bool):
        fn = ('lastOppWonN_' + str(n))
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        source: pd.DataFrame = self.addDatetimeColumns(source)
        cols = [('lastOppWonN_' + str(i)) for i in range(n)]
        cols = [(prefix + col) for prefix in ['home_', 'away_'] for col in cols]
        new_df = pd.DataFrame(columns=list(source.columns)+cols)
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), fn)
            dt, home_abbr, away_abbr = row[['datetime', 'home_abbr', 'away_abbr']]
            home_stats = self.getStats(n, home_abbr, away_abbr, dt)
            away_stats = self.getStats(n, away_abbr, home_abbr, dt)
            new_df.loc[len(new_df.index)] = list(row.values) + home_stats + away_stats
        new_df.drop(columns=['week', 'year', 'datetime'], inplace=True)
        if not isNew:
            self.saveFrame(new_df, fn)
        return new_df
    def saveFrame(self, df: pd.DataFrame, name):
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
    
# END / LastOppWonN

######################

# df = pd.read_csv("%s.csv" % "../../../../data/gameData")
# low = LastOppWonN(df, "./")

# # source = pd.read_csv("%s.csv" % "../source/source")
# source = pd.read_csv("%s.csv" % "../source/new_source")
# low.buildLastOppWonN(5, source, True)

