import pandas as pd
import numpy as np
import os
import datetime

class StarterInfo:
    def __init__(self, _dir):
        self._dir = _dir
        self.s_dir = self._dir + "../../../../starters/"
        self.data_dir = self._dir + "../../../../data/"
        self.si = pd.read_csv("%s.csv" % (self.s_dir + "startsInfo"))
        self.si = self.addDatetimeColumns(self.si)
        self.sdf = pd.read_csv("%s.csv" % (self.s_dir + "allStarters"))
        self.positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def getInfo(self, key, abbr, wy, datetime, isNew: bool):
        df = self.si
        t_cols = df.columns[4:10]
        if not isNew:
            starters = self.sdf.loc[(self.sdf['key']==key)&(self.sdf['abbr']==abbr), 'starters'].values[0]
        else:
            s_path = 'starters_22/' if '2022' in wy else 'starters_23/'
            fn = 'starters_w' + wy.split(" | ")[0]
            sdf = pd.read_csv("%s.csv" % (self.data_dir + s_path + fn))
            try:
                starters = sdf.loc[(sdf['key']==key)&(sdf['abbr']==abbr), 'starters'].values[0]
            except IndexError:
                print(f"Missing starters: {key}, {abbr}")
        info = { pos: [] for pos in self.positions }
        for s in starters.split("|"):
            pid, pos = s.split(":")
            if pos in self.positions:
                try:
                    stats = df.loc[(df['p_id']==pid)&(df['datetime']<datetime), t_cols].values[-1]
                    info[pos].append(list(stats))
                except IndexError:
                    continue
        all_info = []
        for pos in info:
            if len(info[pos]) == 0:
                info[pos].append([np.nan for _ in range(len(t_cols))])
            data = list(zip(*info[pos]))
            [all_info.append(sum(col)/len(col)) for col in data]
        return all_info
    def buildStarterInfo(self, source: pd.DataFrame, isNew: bool):
        if 'starterInfo.csv' in os.listdir(self._dir) and not isNew:
            print('starterInfo.csv already built.')
            return
        print('Creating starterInfo...')
        source = self.addDatetimeColumns(source)
        _types = ['career', 'curr_season', 'last_season']
        cols = [(_type + "_" + suffix) for suffix in ['gamesPlayed', 'starts'] for _type in _types]
        cols = [('starterInfo_' + pos + '_' + col) for pos in self.positions for col in cols]
        cols = [(prefix + col) for prefix in ['home_', 'away_'] for col in cols]
        new_df = pd.DataFrame(columns=['key', 'wy', 'home_abbr', 'away_abbr'] + cols)
        for index, (key, wy, home_abbr, away_abbr, datetime) in enumerate(source[['key', 'wy', 'home_abbr', 'away_abbr', 'datetime']].values):
            self.printProgressBar(index, len(source.index), 'StarterInfo')
            home_info = self.getInfo(key, home_abbr, wy, datetime, isNew)
            away_info = self.getInfo(key, away_abbr, wy, datetime, isNew)
            new_df.loc[len(new_df.index)] = [
                key, wy, home_abbr, away_abbr
            ] + home_info + away_info
        new_df.fillna(new_df.mean(), inplace=True)
        if not isNew:
            self.saveFrame(new_df, (self._dir + "starterInfo"))
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
    
# END / StarterInfo

##############################

# si = StarterInfo("./")

# # source = pd.read_csv("%s.csv" % "../source/source")
# # si.buildStarterInfo(source, False)

# new_source = pd.read_csv("%s.csv" % "../source/new_source")
# si.buildStarterInfo(new_source, True)