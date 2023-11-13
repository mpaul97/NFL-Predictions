import pandas as pd
import numpy as np
import os

class AdvancedStats:
    def __init__(self, adf: pd.DataFrame, _dir):
        self.adf = adf
        self._dir = _dir
        # info
        self.drop_cols = open((self._dir + "dropCols.txt"), "r").read().split("\n")
        return
    def convertRecord(self, record: str):
        if type(record) == float:
            return 0
        wins, loses, ties = record.split("-")
        wins, loses, ties = int(wins), int(loses), int(ties)
        return (2*(wins+ties))/(2*sum([wins, loses, ties]))
    def buildAdvancedStats(self, source: pd.DataFrame, isNew: bool):
        fn = "advancedStats"
        if (fn + ".csv") in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print("Creating " + fn + "...")
        df = self.adf.drop(columns=self.drop_cols)
        stat_cols = [col for col in df.columns if col not in ['player_id', 'season']]
        cols = [('advancedStats_' + col) for col in stat_cols]
        new_df = pd.DataFrame(columns=list(source.columns)+cols)
        for index, row in source.iterrows():
            if not isNew:
                self.printProgressBar(index, len(source.index), fn)
            pid, wy = row[['p_id', 'wy']]
            week, year = int(wy.split(" | ")[0]), int(wy.split(" | ")[1])
            season = str(year-1) if week == 1 else str(year)
            stats: pd.DataFrame = df.loc[(df['player_id']==pid)&(df['season']==season)]
            if not stats.empty:
                # stats.at[stats.index[0], 'qb_record'] = self.convertRecord(stats['qb_record'].values[0])
                stats = stats[stat_cols]
                stats = stats.values[0]
            else:
                stats = np.zeros(len(stat_cols))
            new_df.loc[len(new_df.index)] = list(row.values) + list(stats)
        new_df.fillna(0, inplace=True)
        if not isNew:
            self.saveFrame(new_df, (self._dir + "advancedStats"))
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
    
# END / AdvancedStats

########################

# adf = pd.read_csv("%s.csv" % "../../../../data/advancedStats")
# ads = AdvancedStats(adf, "./")

# # source = pd.read_csv("%s.csv" % "../source/source")
# source = pd.read_csv("%s.csv" % "../source/new_source")
# ads.buildAdvancedStats(source, True)