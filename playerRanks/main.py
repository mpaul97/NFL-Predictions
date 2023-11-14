import pandas as pd
import numpy as np
import os
import datetime
import random

class Main:
    def __init__(self, iteration: int, _dir: str):
        self.iteration = iteration
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.position_dir = self._dir + "../data/positionData/"
        self.snap_dir = self._dir + "../snapCounts/"
        self.positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        self.target_stats = {
            'QB': ['completed_passes', 'passing_touchdowns', 'interceptions_thrown']
        }
        self.merge_cols = ['p_id', 'game_key', 'abbr', 'wy']
        # frames
        self.df: pd.DataFrame = None
        self.all_df: pd.DataFrame = None
        # features
        self.feature_funcs = [
            
        ]
        return
    def get_datetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def add_datetime_columns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.get_datetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def seasonAvgStats_feature(self):
        
        return
    def build_train(self):
        for position in self.positions:
            self.set_df(position)
            df = self.df
        return
    def set_df(self, position: str):
        self.df = pd.read_csv("%s.csv" % (self.position_dir + position + "Data")) if position not in ['DL', 'LB'] else pd.read_csv("%s.csv" % (self.position_dir + "LBDLData"))
        return
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def print_progress_bar(self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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

##################

m = Main(
    iteration=1,
    _dir="./"
)

m.build_train()