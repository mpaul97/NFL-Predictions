import pandas as pd
import numpy as np
import os
import datetime
import multiprocessing
import time

pd.options.mode.chained_assignment = None

class AvgStatsN:
    def __init__(self, data: dict, _dir: str):
        self._dir = _dir
        self.data = data
        self.data = { pos: self.addDatetimeColumns(data[pos]) for pos in data }
        # info
        self.target_stats = { # !!! MUST BE SAME LENGTH !!!
            'QB': ['passing_yards', 'passing_touchdowns'],
            'RB': ['rush_yards', 'rush_touchdowns'],
            'WR': ['receiving_yards', 'receiving_touchdowns'],
            'TE': ['receiving_yards', 'receiving_touchdowns']
        }
        self.str_vals = 'abcdefghijklmnopqrstuvwyxz'
        self.N: int = None
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def buildAvgStatsN(self, n: int, source: pd.DataFrame, isNew: bool):
        fn = "avgStatsN_" + str(n)
        if (fn + '.csv') in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print("Creating " + fn + "...")
        source: pd.DataFrame = self.addDatetimeColumns(source)
        cols = [('avgStatsN_target_' + self.str_vals[j]) for j in range(len(self.target_stats['QB']))]
        new_df = pd.DataFrame(columns=['p_id', 'datetime']+cols)
        for index, (pid, position, datetime) in enumerate(source[['p_id', 'position', 'datetime']].values):
            if not isNew:
                self.printProgressBar(index, len(source.index), fn)
            df: pd.DataFrame = self.data[position]
            total_length = len(self.target_stats[position])
            try:
                stats = df.loc[(df['p_id']==pid)&(df['datetime']<datetime), self.target_stats[position]].values[-n:]
                if len(stats) > 0:
                    stats = np.mean(stats, axis=0)
                else:
                    stats = np.zeros(total_length)
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
    def func(self, source: pd.DataFrame):
        cols = [('avgStatsN_target_' + self.str_vals[j]) for j in range(len(self.target_stats['QB']))]
        new_df = pd.DataFrame(columns=['p_id', 'datetime']+cols)
        for index, (pid, position, datetime) in enumerate(source[['p_id', 'position', 'datetime']].values):
            df: pd.DataFrame = self.data[position]
            total_length = len(self.target_stats[position])
            try:
                stats = df.loc[(df['p_id']==pid)&(df['datetime']<datetime), self.target_stats[position]].values[-self.N:]
                if len(stats) > 0:
                    stats = np.mean(stats, axis=0)
                else:
                    stats = np.zeros(total_length)
            except IndexError:
                stats = np.zeros(total_length)
            if len(stats) < total_length:
                dif = total_length - len(stats)
                stats = np.concatenate((stats, np.zeros(dif)))
            new_df.loc[len(new_df.index)] = [pid, datetime] + list(stats)
        new_df = source.merge(new_df, on=['p_id', 'datetime'])
        new_df.drop(columns=['week', 'year', 'datetime'], inplace=True)
        return new_df
    def buildAvgStatsN_parallel(self, n: int, source: pd.DataFrame):
        fn = "avgStatsN_" + str(n)
        if (fn + '.csv') in os.listdir(self._dir):
            print(fn + " already exists.")
            return
        print("Creating " + fn + "...")
        self.N = n
        source: pd.DataFrame = self.addDatetimeColumns(source)
        num_cores = multiprocessing.cpu_count()-1
        num_partitions = num_cores
        source_split = np.array_split(source, num_partitions)
        df_list = []
        if __name__ in ['__main__', 'fantasyPredictions.features.avgStatsN.avgStatsN']:
            pool = multiprocessing.Pool(num_cores)
            all_dfs = pd.concat(pool.map(self.func, source_split))
            df_list.append(all_dfs)
            pool.close()
            pool.join()
        if df_list:
            new_df = pd.concat(df_list)
            self.saveFrame(new_df, (self._dir + fn))
        return
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
    
# END / AvgStatsN

#########################

# start = time.time()

# data_dir = "../../../../data/positionData/"
# positions = ['QB', 'RB', 'WR', 'TE']
# data = { pos: pd.read_csv("%s.csv" % (data_dir + pos + "Data")) for pos in positions }

# asn = AvgStatsN(data, "./")

# source = pd.read_csv("%s.csv" % "../source/source")
# # asn.buildAvgStatsN(5, source, False)
# asn.buildAvgStatsN_parallel(5, source)

# if __name__ == '__main__':
#     end = time.time()
#     elapsed = end - start
#     print(f"Time elapsed: {elapsed}")