import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../')

from paths import DATA_PATH, POSITION_PATH, STARTERS_PATH, TEAMNAMES_PATH, COACHES_PATH, MADDEN_PATH, NAMES_PATH, SNAP_PATH, PLAYER_RANKS_PATH, PBP_PATH

# from gamePredictions.build import Build
# from fantasyPredictions.build import Build as FpBuild
# from database import Firebase

class Main:
    def __init__(self, week, year):
        self.all_paths = {
            'dp': DATA_PATH, 'pp': POSITION_PATH, 'sp': STARTERS_PATH,
            'tnp': TEAMNAMES_PATH, 'cp': COACHES_PATH, 'mrp': MADDEN_PATH,
            'sc': SNAP_PATH, 'pr': PLAYER_RANKS_PATH, 'pbp': PBP_PATH
        }
        self.week = week
        self.year = year
        self.gp_dir = 'gamePredictions/'
        self.fp_dir = 'fantasyPredictions/'
        return
    # game predictions + points predictions
    def gpPredicitions(self, name):
        if name == 'train':
            b = Build(self.all_paths, self.gp_dir)
            b.main()
        elif name == 'new':
            b = Build(self.all_paths, self.gp_dir)
            b.new_main(self.week, self.year)
            # tfp = TfPredict(self.gp_dir)
            # tfp.build()
            # tfp.buildConsensus()
        else:
            b = Build(self.all_paths, self.gp_dir)
            b.main()
            b.new_main(self.week, self.year)
            # tfp = TfPredict(self.gp_dir)
            # tfp.build()
            # tfp.buildConsensus()
        return
    # fantasy points + rank predictions
    def fpPredicitions(self, name):
        if name == 'train':
            b = FpBuild(self.all_paths, self.fp_dir)
            b.main()
        elif name == 'new':
            b = FpBuild(self.all_paths, self.fp_dir)
            b.new_main(self.week, self.year)
            # tfp = FpTfPredict(self.fp_dir)
            # tfp.build()
        else:
            b = FpBuild(self.all_paths, self.fp_dir)
            b.main()
            b.new_main(self.week, self.year)
            # tfp = FpTfPredict(self.fp_dir)
            # tfp.build()
        return
    # find missing columns
    def findMissingCols_gp(self):
        train = pd.read_csv("%s.csv" % (self.gp_dir + "train"))
        test = pd.read_csv("%s.csv" % (self.gp_dir + "test"))
        # for col in train.columns:
        #     if col not in test.columns:
        #         print(col)
        print(test.isna().any())
        return
    def test(self):
        b = Build(self.all_paths, self.gp_dir)
        # b.test_func()
        b.buildPredTargets()
        return

########################

# if __name__ == '__main__':
#     m = Main(
#         week=13,
#         year=2023
#     )
#     m.gpPredicitions('new')
    # m.fpPredicitions('both')
    # m.findMissingCols_gp()
    # m.test()
    # m.createUiData()
        
from gamePredictions.build_v2 import Build

b = Build()
b.main()