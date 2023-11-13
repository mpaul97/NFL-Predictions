import pandas as pd
import numpy as np
import os
import datetime

class PositionGroupSeasonAvgSnapPcts:
    def __init__(self, df: pd.DataFrame, sdf: pd.DataFrame, _dir):
        self.df: pd.DataFrame = self.addDatetimeColumns(df)
        self.sdf = sdf
        self._dir = _dir
        # params
        self.groups = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        self.off_groups = ['QB', 'RB', 'WR', 'TE', 'OL']
        return
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getStats(self, key: str, wy: str, abbr: str):
        if wy == '1 | 2012':
            return [np.nan for _ in range(len(self.groups))]
        df = self.df
        week, year = [int(w) for w in wy.split(" | ")]
        starters = (self.sdf.loc[(self.sdf['key']==key)&(self.sdf['abbr']==abbr), 'starters'].values[0]).split("|")
        all_stats = []
        for position in self.groups:
            pct_col = 'off_pct' if position in self.off_groups else 'def_pct'
            players = [p.split(":")[0] for p in starters if position in p]
            season = (year-1) if week == 1 else year
            all_pcts = []
            for p in players:
                stats = df.loc[(df['p_id']==p)&(df['year']==season), pct_col].values
                if len(stats) != 0:
                    all_pcts.append(np.mean(stats))
                else:
                    all_pcts.append(0)
            all_stats.append(np.mean(all_pcts) if len(all_pcts) != 0 else 0)
        return all_stats
    def buildPositionGroupSeasonAvgSnapPcts(self, source: pd.DataFrame, isNew: bool):
        fn = "positionGroupSeasonAvgSnapPcts"
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        if not isNew:
            start = source.loc[source['wy'].str.contains('2012')].index.values[0]
            source = source.loc[source.index>=start]
            source = source.reset_index(drop=True)
        cols = [(position + '_' + fn) for position in self.groups]
        cols = [(prefix + col) for prefix in ['home_', 'away_'] for col in cols]
        new_df = pd.DataFrame(columns=list(source.columns)+cols)
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), fn)
            key, wy, home_abbr, away_abbr = row[['key', 'wy', 'home_abbr', 'away_abbr']]
            home_stats = self.getStats(key, wy, home_abbr)
            away_stats = self.getStats(key, wy, away_abbr)
            new_df.loc[len(new_df.index)] = list(row.values) + home_stats + away_stats
        new_df.fillna(new_df.mean(), inplace=True)
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
    
# END / PositionGroupSeasonAvgSnapPcts

########################

# sdf = pd.read_csv("%s.csv" % "../../../../starters/allStarters")
# df = pd.read_csv("%s.csv" % "../../../../snapCounts/snap_counts")
# pgsp = PositionGroupSeasonAvgSnapPcts(df, sdf, "./")

# source = pd.read_csv("%s.csv" % "../source/source")
# pgsp.buildPositionGroupSeasonAvgSnapPcts(source, isNew=False)
