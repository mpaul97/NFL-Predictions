import pandas as pd
import numpy as np
import os
import datetime

class LastGameStatsN:
    def __init__(self, df: pd.DataFrame, _dir):
        self.df = df
        self.df: pd.DataFrame = self.addDatetimeColumns(self.df)
        self._dir = _dir
        self.stat_cols = [
            'net_pass_yards', 'rush_attempts', 'total_yards',
            'turnovers', 'first_downs', 'times_sacked'
        ]
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def flatten(self, l):
        return [item for sublist in l for item in sublist]
    def replaceCols(self, df: pd.DataFrame, isHome: bool):
        if isHome:
            col_names = list(df.columns)
            for name in col_names:
                new_col_name = name.replace('home_', '').replace('away_', 'opp_')
                df = df.rename(columns={name: new_col_name})
        else:
            col_names = list(df.columns)
            for name in col_names:
                new_col_name = name.replace('away_', '').replace('home_', 'opp_')
                df = df.rename(columns={name: new_col_name})
        return df
    def getStats(self, n: int, dt: datetime, abbr: str):
        stats = self.df.loc[
            ((self.df['home_abbr']==abbr)|(self.df['away_abbr']==abbr))&
            (self.df['datetime']<dt)
        ].tail(n)
        cols = [col for col in stats.columns if 'home'in col or 'away' in col]
        stats = stats[cols]
        stat_cols = self.stat_cols + [('opp_' + col) for col in self.stat_cols]
        length = len(stat_cols)*n
        try:
            stats['isHome'] = stats.apply(lambda x: 1 if x['home_abbr']==abbr else 0, axis=1)
            stats.drop(columns=['home_abbr', 'away_abbr'], inplace=True)
            home_stats = self.replaceCols(stats.loc[stats['isHome']==1].drop(columns=['isHome']), True)
            away_stats = self.replaceCols(stats.loc[stats['isHome']==0].drop(columns=['isHome']), False)
            all_stats = pd.concat([home_stats, away_stats])[stat_cols]
            all_stats = all_stats.iloc[::-1]
            all_stats = self.flatten(all_stats.apply(lambda x: x.tolist(), axis=1).tolist())
        except ValueError: # empty stats
            all_stats = [np.nan for _ in range(length)]
        if len(all_stats) < length: # not enough games
            dif = length - len(all_stats)
            nan_list = [np.nan for _ in range(dif)]
            all_stats += nan_list
        return all_stats
    def buildLastGameStatsN(self, n: int, source: pd.DataFrame, isNew: bool):
        fn = ('lastGameStatsN_' + str(n))
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        source: pd.DataFrame = self.addDatetimeColumns(source)
        cols = self.stat_cols + [('opp_' + col) for col in self.stat_cols]
        cols = [(col + '_' + str(i)) for i in range(n) for col in cols]
        cols = [(prefix + col) for prefix in ['home_', 'away_'] for col in cols]
        cols = [('lastGameStatsN_' + col) for col in cols]
        new_df = pd.DataFrame(columns=list(source.columns)+cols)
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), fn)
            dt, home_abbr, away_abbr = row[['datetime', 'home_abbr', 'away_abbr']]
            home_stats = self.getStats(n, dt, home_abbr)
            away_stats = self.getStats(n, dt, away_abbr)
            new_df.loc[len(new_df.index)] = list(row.values) + home_stats + away_stats
        new_df.drop(columns=['week', 'year', 'datetime'], inplace=True)
        new_df.fillna(new_df.mean()*0.8, inplace=True)
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
    
# END / LastGameStatsN

#############################

# df = pd.read_csv("%s.csv" % "../../../../data/gameData")
# lgs = LastGameStatsN(df, "./")

# source = pd.read_csv("%s.csv" % "../source/source")
# # source = pd.read_csv("%s.csv" % "../source/new_source")
# lgs.buildLastGameStatsN(5, source, False)