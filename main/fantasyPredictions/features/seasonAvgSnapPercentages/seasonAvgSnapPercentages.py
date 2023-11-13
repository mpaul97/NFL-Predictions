import pandas as pd
import numpy as np
import os
import datetime

class SeasonAvgSnapPercentages:
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
    def build(self, source: pd.DataFrame, isNew: bool):
        fn = "seasonAvgSnapPercentages"
        if (fn + ".csv") in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print("Creating " + fn + "...")
        source = self.addDatetimeColumns(source)
        df = self.addDatetimeColumns(self.df)
        new_df = pd.DataFrame(columns=list(source.columns)+[fn])
        for index, row in source.iterrows():
            if not isNew:
                self.printProgressBar(index, len(source.index), fn)
            pid, wy, dt, week, year = row[['p_id', 'wy', 'datetime', 'week', 'year']]
            if wy != df['wy'].values[0]:
                season = (year - 1) if (week == 1) else year
                stats = df.loc[(df['p_id']==pid)&(df['datetime']<dt)&(df['year']==season), 'off_pct'].values
                mean = np.mean(stats) if len(stats) != 0 else 0
            else:
                mean = np.nan
            new_df.loc[len(new_df.index)] = list(row.values) + [mean]
        new_df.drop(columns=['week', 'year', 'datetime'], inplace=True)
        new_df.fillna(new_df.mean(), inplace=True)
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
    
# END / SeasonAvgSnapPercentages

##########################

# df = pd.read_csv("%s.csv" % "../../../../snapCounts/snap_counts")
# sasp = SeasonAvgSnapPercentages(df, "./")

# # source = pd.read_csv("%s.csv" % "../source/source")
# source = pd.read_csv("%s.csv" % "../source/new_source")
# sasp.build(source, True)