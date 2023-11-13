import pandas as pd
import numpy as np
import os
import datetime

class SummariesAvgN:
    def __init__(self, df: pd.DataFrame, _dir):
        self.df = df
        self.df: pd.DataFrame = self.addDatetimeColumns(self.df)
        self._dir = _dir
        self.stat_cols = {
            'home': [('home_' + str(i)) for i in range(1, 5)],
            'away': [('away_' + str(i)) for i in range(1, 5)]
        }
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def getStats(self, n: int, dt: datetime, abbr: str):
        stats = self.df.loc[
            (self.df['datetime']<dt)&
            ((self.df['home_abbr']==abbr)|(self.df['away_abbr']==abbr))
        ].tail(n)
        if not stats.empty:
            # for stats
            home_stats = stats.loc[stats['home_abbr']==abbr, self.stat_cols['home']]
            away_stats = stats.loc[stats['away_abbr']==abbr, self.stat_cols['away']]
            home_avgs = home_stats.mean().values
            away_avgs = away_stats.mean().values
            for_avgs = np.mean(np.vstack((home_avgs, away_avgs)), axis=0)
            # allowed stats
            a_home_stats = stats.loc[stats['home_abbr']==abbr, self.stat_cols['away']]
            a_away_stats = stats.loc[stats['away_abbr']==abbr, self.stat_cols['home']]
            a_home_avgs = a_home_stats.mean().values
            a_away_avgs = a_away_stats.mean().values
            allowed_avgs = np.mean(np.vstack((a_home_avgs, a_away_avgs)), axis=0)
        else:
            for_avgs = [np.nan for _ in range(4)]
            allowed_avgs = [np.nan for _ in range(4)]
        return list(for_avgs) + list(allowed_avgs)
    def buildSummariesAvgN(self, n: int, source: pd.DataFrame, isNew: bool):
        """
        Last N average points per quarter and
        last N average allowed points per quarter,
        no OT
        Args:
            n (int): _description_
            source (pd.DataFrame): _description_
        """
        fn = "summariesAvgN_" + str(n)
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        source: pd.DataFrame = self.addDatetimeColumns(source)
        cols = [('summariesAvgN-' + str(n) + '_' + prefix + str(i)) for prefix in ['for_', 'allowed_'] for i in range(1, 5)]
        cols = [(prefix + col) for prefix in ['home_', 'away_'] for col in cols]
        new_df = pd.DataFrame(columns=['home_abbr', 'away_abbr', 'datetime']+cols)
        for index, (dt, home_abbr, away_abbr) in enumerate(source[['datetime', 'home_abbr', 'away_abbr']].values):
            self.printProgressBar(index, len(source.index), fn)
            home_stats = self.getStats(n, dt, home_abbr)
            away_stats = self.getStats(n, dt, away_abbr)
            new_df.loc[len(new_df.index)] = [home_abbr, away_abbr, dt] + home_stats + away_stats
        source = source.merge(new_df, on=['home_abbr', 'away_abbr', 'datetime'])
        source.drop(columns=['week', 'year', 'datetime'], inplace=True)
        source.fillna(source.mean(), inplace=True)
        if not isNew:
            self.saveFrame(source, fn)
        return source
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
    
# END / SummariesAvgN

#########################

# df = pd.read_csv("%s.csv" % "../../../../data/summaries")
# san = SummariesAvgN(df, "./")

# # source = pd.read_csv("%s.csv" % "../source/source")
# source = pd.read_csv("%s.csv" % "../source/new_source")
# cd = san.buildSummariesAvgN(5, source, True)