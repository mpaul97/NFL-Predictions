import pandas as pd
import numpy as np
import os

pd.options.mode.chained_assignment = None

class Expected:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.df = pd.read_csv("%s.csv" % (self._dir + "snap_counts"))
        self.sp = pd.read_csv("%s.csv" % (self.data_dir + "snap_positions"))
        self.merge_cols = ['key', 'wy', 'abbr', 'p_id', 'position']
        self.off_positions = ['QB', 'RB', 'WR', 'TE', 'OL']
        return
    def get_position(self, position: str):
        return self.sp.loc[self.sp['position']==position, 'simplePosition'].values[0]
    def build_target(self):
        start = self.df.loc[self.df['wy'].str.contains('2013')].index.values[0]
        df: pd.DataFrame = self.df.loc[self.df.index>=start]
        df['position'] = df['position'].apply(lambda x: self.get_position(x))
        off_df = df.loc[df['position'].isin(self.off_positions)]
        off_df = off_df[self.merge_cols+['off_pct']]
        off_df.columns = self.merge_cols + ['pct']
        def_df = df.loc[~df['position'].isin(self.off_positions)]
        def_df = def_df[self.merge_cols+['def_pct']]
        def_df.columns = self.merge_cols + ['pct']
        new_df = pd.concat([off_df, def_df])
        new_df = new_df.loc[~new_df['position'].isin(['K', 'P', 'LS'])]
        new_df.sort_values(by=['key'], inplace=True)
        self.save_frame(new_df, (self.data_dir + "expected_target"))
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
    
# END / Expected

#####################

e = Expected("./")

e.build_target()