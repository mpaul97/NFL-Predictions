# projects player with volume percentage below 1.0 if they played full game
import pandas as pd
import numpy as np
import os

class VolumeProjections:
    def __init__(self, _dir):
        self.positions = ['QB', 'RB', 'WR', 'TE']
        self._dir = _dir
        self.data_dir = _dir + "../positionData/"
        self.proj_dir = _dir + "projectionData/"
        self.position_frames = { pos: pd.read_csv("%s.csv" % (self.data_dir + pos + "Data")) for pos in self.positions }
        self.target_cols = {
            'QB': ['passing_yards', 'passing_touchdowns', 'rush_yards', 'rush_touchdowns', 'interceptions_thrown'],
            'RB': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
            'WR': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
            'TE': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
        }
        self.str_cols = ['p_id', 'game_key', 'wy', 'position']
        return
    def build(self):
        for pos in self.positions:
            df: pd.DataFrame = self.position_frames[pos]
            df = df.loc[df['volume_percentage'].between(0.1, 0.95)]
            t_cols = self.target_cols[pos]
            df = df[self.str_cols+t_cols+['volume_percentage']]
            new_df = pd.DataFrame(columns=self.str_cols+t_cols)
            for index, vals in enumerate(df[self.str_cols+t_cols+['volume_percentage']].values):
                self.printProgressBar(index, len(df.index), (pos + "-volumeProjections"))
                vol = vals[-1]
                vol_dif = 1.0 - vol
                multi = vol_dif*2
                str_vals = list(vals[:len(self.str_cols)])
                vals = vals[len(self.str_cols):-1]
                func = np.vectorize(lambda x: (x*multi)+x)
                new_vals = func(vals)
                new_df.loc[len(new_df.index)] = str_vals + list(new_vals)
            self.saveFrame(new_df, (self.proj_dir + pos + "_volume_projections"))
        return
    def test(self, pos: str):
        df = pd.read_csv("%s.csv" % (self.proj_dir + pos + "_volume_projections"))
        df = df.loc[df['wy'].str.contains('2022')]
        df.sort_values(by=self.target_cols[pos][0], ascending=False, inplace=True)
        print(df)
        return
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
    
#######################

vp = VolumeProjections('./')

# vp.build()

vp.test('TE')