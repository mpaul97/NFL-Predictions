import pandas as pd
import numpy as np
import os
import datetime

class LastStatsN:
    def __init__(self, data: dict, _dir: str):
        self._dir = _dir
        self.data = data
        self.data = { pos: self.addDatetimeColumns(data[pos]) for pos in data }
        # info
        self.target_stats = { # !!! MUST BE SAME LENGTH !!!
            'QB': [
                'passing_yards', 'passing_touchdowns', 'quarterback_rating', 
                'interceptions_thrown', 'td_percentage'
            ],
            'RB': [
                'rush_yards', 'rush_touchdowns', 'receptions',
                'receiving_yards', 'rush_yards_per_attempt'
            ],
            'WR': [
                'receiving_yards', 'receiving_touchdowns', 'receptions',
                'times_pass_target', 'touchdown_per_touch'
            ],
            'TE': [
                'receiving_yards', 'receiving_touchdowns', 'receptions',
                'times_pass_target', 'touchdown_per_touch'
            ],
        }
        self.str_vals = 'abcdefghijklmnopqrstuvwyxz'
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def buildLastStatsN(self, n: int, source: pd.DataFrame, isNew: bool):
        fn = "lastStatsN_" + str(n)
        if (fn + '.csv') in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print("Creating " + fn + "...")
        source: pd.DataFrame = self.addDatetimeColumns(source)
        cols = [('lastStatsN_target_' + self.str_vals[j] + '-' + str(i)) for i in range(n) for j in range(len(self.target_stats['QB']))]
        new_df = pd.DataFrame(columns=['p_id', 'datetime']+cols)
        for index, (pid, position, datetime) in enumerate(source[['p_id', 'position', 'datetime']].values):
            if not isNew:
                self.printProgressBar(index, len(source.index), fn)
            df: pd.DataFrame = self.data[position]
            total_length = n*len(self.target_stats[position])
            try:
                stats = df.loc[(df['p_id']==pid)&(df['datetime']<datetime), self.target_stats[position]].values[-n:]
                stats = np.flip(stats, axis=0).flatten()
            except IndexError:
                stats = np.zeros(total_length)
            if len(stats) < total_length:
                dif = total_length - len(stats)
                stats = np.concatenate((stats, np.zeros(dif)))
            new_df.loc[len(new_df.index)] = [pid, datetime] + list(stats)
        new_df = source.merge(new_df, on=['p_id', 'datetime'])
        new_df.drop(columns=['week', 'year', 'datetime'], inplace=True)
        if not isNew:
            self.saveFrame(new_df, (self._dir + fn))
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
    
# END / LastStatsN

#########################

# data_dir = "../../../../data/positionData/"
# positions = ['QB', 'RB', 'WR', 'TE']
# data = { pos: pd.read_csv("%s.csv" % (data_dir + pos + "Data")) for pos in positions }

# lsn = LastStatsN(data, "./")

# source = pd.read_csv("%s.csv" % "../source/source")
# lsn.buildLastStatsN(10, source, False)