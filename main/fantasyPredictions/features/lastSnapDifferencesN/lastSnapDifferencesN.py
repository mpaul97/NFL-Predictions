import pandas as pd
import numpy as np
import os
import datetime

class LastSnapDifferencesN:
    def __init__(self, df: pd.DataFrame, _dir):
        self.df = df
        self._dir = _dir
        return
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def build(self, n: int, source: pd.DataFrame, isNew: bool):
        fn = "lastSnapDifferencesN_" + str(n-1)
        if (fn + ".csv") in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print("Creating " + fn + "...")
        source = self.addDatetimeColumns(source)
        df = self.addDatetimeColumns(self.df)
        info = df[['abbr', 'wy', 'datetime']].drop_duplicates()
        cols = [(fn + "-" + str(i)) for i in range(n-1)]
        new_df = pd.DataFrame(columns=list(source.columns)+cols)
        for index, row in source.iterrows():
            if not isNew:
                self.printProgressBar(index, len(source.index), fn)
            pid, abbr, wy, dt, week, year = row[['p_id', 'abbr', 'wy', 'datetime', 'week', 'year']]
            if wy != '1 | 2012':
                wys = info.loc[(info['abbr']==abbr)&(info['datetime']<dt), ['wy']].tail(n)
                stats = df.loc[(df['p_id']==pid)&(df['wy'].isin(wys['wy'].values)), ['p_id', 'wy', 'off_pct']]
                stats = wys.merge(stats, on=['wy'], how='left')
                stats = stats['off_pct'].values
                stats = np.nan_to_num(stats)
                if len(stats) < n:
                    dif = n - len(stats)
                    stats = np.concatenate((stats, np.zeros(dif)))
                stats = np.flip(np.ediff1d(stats))
            else:
                stats = np.array([np.nan for _ in range(n-1)])
            new_df.loc[len(new_df.index)] = list(row.values) + list(stats)
        new_df.drop(columns=['week', 'year', 'datetime'], inplace=True)
        new_df.fillna(new_df.mean(), inplace=True)
        new_df = new_df.round(4)
        if not isNew:
            self.saveFrame(new_df, (self._dir + fn))
        return new_df
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
    
# END / LastSnapDifferencesN

#############################

# df = pd.read_csv("%s.csv" % "../../../../snapCounts/snap_counts")
# lsp = LastSnapDifferencesN(df, "./")

# source = pd.read_csv("%s.csv" % "../source/source")
# # source = pd.read_csv("%s.csv" % "../source/new_source")
# lsp.build(11, source, False)
