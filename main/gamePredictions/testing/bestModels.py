import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../../../')

from paths import DATA_PATH, POSITION_PATH, STARTERS_PATH, TEAMNAMES_PATH, COACHES_PATH, MADDEN_PATH, NAMES_PATH, SNAP_PATH
from main.gamePredictions.build import Build

class BestModels:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.cd = pd.read_csv("%s.csv" % "../../../data/gameData")
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        self.all_paths = {
            'dp': DATA_PATH, 'pp': POSITION_PATH, 'sp': STARTERS_PATH,
            'tnp': TEAMNAMES_PATH, 'cp': COACHES_PATH, 'mrp': MADDEN_PATH,
            'sc': SNAP_PATH
        }
        self.gp_dir = 'gamePredictions/'
        return
    def func(self, row, col):
        winning_abbr = row['home_abbr'] if row[col] == 1 else row['away_abbr']
        return 1 if row['winning_abbr'] == winning_abbr else 0
    def find(self):
        df = pd.concat([pd.read_csv(self.data_dir + fn) for fn in os.listdir(self.data_dir) if '.csv' in fn])
        won_cols = [col for col in df.columns if 'won' in col or col == 'most_common']
        df = df[self.str_cols+won_cols]
        cd = self.cd[['key', 'winning_abbr']]
        df = df.merge(cd, on=['key'])
        for col in won_cols:
            df['correct_'+ col] = df.apply(lambda x: self.func(x, col), axis=1)
        new_df = pd.DataFrame(columns=['name', 'total', 'percentage'])
        for col in [col for col in df.columns if 'correct' in col]:
            total = sum(df[col].values)
            length = len(df.index)
            new_df.loc[len(new_df.index)] = [col.replace('correct_',''), total, round((total/length), 2)]
        new_df.sort_values(by=['percentage'], ascending=False, inplace=True)
        print(new_df)
        return
    
# END / BestModels

########################

bm = BestModels("./")

bm.find()