import pandas as pd
import numpy as np
import os
import datetime

class LastPossessionEpasN:
    def __init__(self, n: int, df: pd.DataFrame, _dir: str):
        self.n = n
        self.df = df
        self.df: pd.DataFrame = self.addDatetimeColumns(self.df)
        self._dir = _dir
        self.dt_cols = ['week', 'year', 'datetime']
        self.stat_cols = ['home_epa_added', 'away_epa_added', 'home_total_epa', 'away_total_epa']
        self.cols = ['off_epa_added', 'def_epa_added', 'off_total_epa', 'def_total_epa']
        self.new_cols = [f'lastPossessionEpasN_{self.n}_{c}_{i}' for i in range(self.n) for c in self.cols]
        return
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getStats(self, abbr: str, dt):
        stats: pd.DataFrame = self.df.loc[
            (self.df['datetime']<dt)&
            ((self.df['home_abbr']==abbr)|(self.df['away_abbr']==abbr))
        ]
        home_stats = stats.loc[stats['home_abbr']==abbr]
        home_stats.drop(columns=['away_abbr'], inplace=True)
        home_stats.columns = ['key', 'wy', 'abbr', 'off_epa_added', 'def_epa_added', 'off_total_epa', 'def_total_epa'] + self.dt_cols
        away_stats = stats.loc[stats['away_abbr']==abbr]
        away_stats.drop(columns=['home_abbr'], inplace=True)
        away_stats.columns = ['key', 'wy', 'abbr', 'def_epa_added', 'off_epa_added', 'def_total_epa', 'off_total_epa'] + self.dt_cols
        all_stats = pd.concat([home_stats, away_stats])
        all_stats.sort_values(by=['datetime'], ascending=False, inplace=True)
        all_stats = all_stats.head(self.n)[self.cols]
        return list(all_stats.values.flatten())
    def func(self, row: pd.Series):
        home_abbr, away_abbr = row[['home_abbr', 'away_abbr']]
        dt = row['datetime']
        home_stats = self.getStats(home_abbr, dt)
        away_stats = self.getStats(away_abbr, dt)
        vals = home_stats + away_stats
        cols = [f'{prefix}_{col}' for prefix in ['home', 'away'] for col in self.new_cols]
        return { col: vals[i] for i, col in enumerate(cols) }
    def build(self, source: pd.DataFrame, isNew: bool):
        fn = f"lastPossessionEpasN_{self.n}"
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        if not isNew:
            start = source.loc[source['wy'].str.contains('2012')].index.values[0]
            source = source.loc[source.index>=start]
            source = source.reset_index(drop=True)
        source = self.addDatetimeColumns(source)
        cols = [f'{prefix}_{col}' for prefix in ['home', 'away'] for col in self.new_cols]
        source[cols] = source.apply(lambda row: self.func(row), axis='columns', result_type='expand')
        source.drop(columns=self.dt_cols, inplace=True)
        if not isNew:
            self.saveFrame(source, (self._dir + fn))
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
    
# END / PossessionEpas

#########################

# df = pd.read_csv("%s.csv" % "../../../../playByPlay_v2/data/features/possession_epas")
# sape = LastPossessionEpasN(5, df, "./")

# source = pd.read_csv("%s.csv" % "../source/new_source")
# df = sape.build(source, True)

# sape.saveFrame(df, "temp")