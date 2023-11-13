import pandas as pd
import numpy as np
import os

class IsStarter:
    def __init__(self, _dir):
        self._dir = _dir
        return
    def buildIsStarter(self, source: pd.DataFrame, sdf: pd.DataFrame, isNew: bool):
        if 'isStarter.csv' in os.listdir(self._dir) and not isNew:
            print('isStarter already exists.')
            return
        print('Creating isStarter...')
        all_starters = []
        for index, row in source.iterrows():
            if not isNew:
                self.printProgressBar(index, len(source.index), 'isStarter')
            key, abbr, pid = row[['key', 'abbr', 'p_id']]
            starters = sdf.loc[(sdf['key']==key)&(sdf['abbr']==abbr), 'starters'].values[0]
            all_starters.append(1 if pid in starters else 0)
        source['isStarter'] = all_starters
        if not isNew:
            self.saveFrame(source, (self._dir + "isStarter"))
        return source
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def printProgressBar (self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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

# END / IsStarter

#######################

# iss = IsStarter("./")

# source = pd.read_csv("%s.csv" % "../source/source")
# sdf = pd.read_csv("%s.csv" % "../../../../starters/allStarters")

# iss.buildIsStarter(source, sdf, False)