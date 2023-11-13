import pandas as pd
import numpy as np
import os
import datetime

pd.options.mode.chained_assignment = None

class Main:
    def __init__(self, _dir):
        self._dir = _dir
        self.position_path = self._dir + "../../data/positionData/"
        self.positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        # frames
        self.mr = pd.read_csv("%s.csv" % (self._dir + "../../maddenRatings/playerRatings"))
        self.position_frames: dict = {
            pos: (pd.read_csv("%s.csv" % (self.position_path + pos + "Data")) if pos not in ['LB', 'DL'] else pd.read_csv("%s.csv" % (self.position_path + "LBDLData")))
            for pos in self.positions
        }
        self.train_stats = {
            'QB': ['passing_yards', 'passing_touchdowns']
        }
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def buildTrain(self):
        """
        Last season and career stats
        For each year of MaddenRatings prior to week 1 of that year
        """
        # for position in self.position_frames:
        position = 'QB'
        df: pd.DataFrame = self.position_frames[position]
        df = df[['p_id', 'wy']+self.train_stats[position]]
        df = self.addDatetimeColumns(df)
        mr = self.mr.loc[(self.mr['year']==2023)&(self.mr['abbr']=='GNB')]
        
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
    
# END / Main

#######################

m = Main("./")

m.buildTrain()
    