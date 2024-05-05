import pandas as pd
import numpy as np
import os

class PlayerRanks:
    def __init__(self, pr_df: pd.DataFrame, _dir: str):
        self.pr_df = pr_df
        self._dir = _dir
        return
    def build(self, source: pd.DataFrame, isNew: bool):
        fn = "playerRanks"
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        if not isNew:
            start = source.loc[source['wy'].str.contains('2012')].index.values[0]
            source = source.loc[source.index>=start]
            source = source.reset_index(drop=True)
        home_df = self.pr_df.copy()
        home_df.columns = ['wy'] + ['home_' + col for col in home_df.columns if col != 'wy']
        source = source.merge(home_df, on=['wy', 'home_abbr'], how='left')
        away_df = self.pr_df.copy()
        away_df.columns = ['wy'] + ['away_' + col for col in away_df.columns if col != 'wy']
        source = source.merge(away_df, on=['wy', 'away_abbr'], how='left')
        source.fillna(source.mean(), inplace=True)
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
    
# END / PlayerRanks

#########################

# pr_df = pd.read_csv("%s.csv" % "../../../../playerRanks/playerRanks_features")
# pr = PlayerRanks(pr_df, "./")

# source = pd.read_csv("%s.csv" % "../source/source")
# # source = pd.read_csv("%s.csv" % "../source/new_source")
# pr.build(source, False)