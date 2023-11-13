import pandas as pd
import numpy as np
import os
import sys
sys.path.append("../")

from paths import DATA_PATH, POSITION_PATH, STARTERS_PATH, TEAMNAMES_PATH, COACHES_PATH, MADDEN_PATH, NAMES_PATH, SNAP_PATH
from gamePredictions.build import Build

class BestModels:
    def __init__(self):
        self.all_paths = {
            'dp': DATA_PATH, 'pp': POSITION_PATH, 'sp': STARTERS_PATH,
            'tnp': TEAMNAMES_PATH, 'cp': COACHES_PATH, 'mrp': MADDEN_PATH,
            'sc': SNAP_PATH
        }
        self.gp_dir = 'gamePredictions/'
        self.test_dir = self.gp_dir + 'testing/data/'
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        return
    def most_common(self, lst):
        return max(set(lst), key=lst.count)
    def func(self, row, col):
        winning_abbr = row['home_abbr'] if row[col] == 1 else row['away_abbr']
        return 1 if row['winning_abbr'] == winning_abbr else 0
    def get_predictions(self, start_week: int, end_week: int, year: int):
        df_list = []
        for i in range(start_week, end_week):
            print('------------------------------------')
            print(f"Week: {i}")
            print('------------------------------------')
            b = Build(self.all_paths, self.gp_dir)
            b.new_main(i, year)
            df_list.append(pd.read_csv("%s.csv" % (self.gp_dir + "predictions")))
        return pd.concat(df_list)
    def show_best_models(self):
        df = pd.read_csv("%s.csv" % (self.test_dir + "all_predictions"))
        cd = pd.read_csv("%s.csv" % (self.all_paths['dp'] + "gameData"))
        won_cols = [col for col in df.columns if 'won' in col]
        df = df[self.str_cols+won_cols]
        df['most_common'] = df.apply(lambda x: self.most_common(list(x[won_cols])), axis=1)
        cd = cd[['key', 'winning_abbr']]
        df = df.merge(cd, on=['key'])
        for col in won_cols+['most_common']:
            df['correct_'+ col] = df.apply(lambda x: self.func(x, col), axis=1)
        new_df = pd.DataFrame(columns=['name', 'total', 'percentage'])
        for col in [col for col in df.columns if 'correct' in col]:
            total = sum(df[col].values)
            length = len(df.index)
            new_df.loc[len(new_df.index)] = [col.replace('correct_',''), total, round((total/length), 2)]
        new_df.sort_values(by=['percentage'], ascending=False, inplace=True)
        print(new_df)
        return
    def find_all(self, week: int, year: int):
        """
        Creates predictions for every week prior,
        overwrites current predictions
        Args:
            week (int): current week
            year (int): current year
        """
        if "all_predictions.csv" in os.listdir(self.test_dir):
            df = pd.read_csv("%s.csv" % (self.test_dir + "all_predictions"))
            weeks = set([int(wy.split(" | ")[0]) for wy in df['wy']])
            missing_weeks = list(weeks.difference(set([i for i in range(1, week)])))
            if len(missing_weeks) != 0:
                print("Updating for weeks: ", missing_weeks)
                new_df = self.get_predictions(min(missing_weeks), week, year)
                new_df = pd.concat([df, new_df])
                self.save_frame(new_df, (self.test_dir +  "all_predictions"))
            else:
                override = input('Override? (y/n): ')
                if override == 'y':
                    new_df = self.get_predictions(1, week, year)
                    self.save_frame(new_df, (self.test_dir +  "all_predictions"))
        else:
            new_df = self.get_predictions(1, week, year)
            self.save_frame(new_df, (self.test_dir +  "all_predictions"))
        self.show_best_models()
        return
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" %  name, index=False)
        return
    
# END / BestModels

######################

bm = BestModels()

bm.find_all(
    week=10, # current week -> get all weeks prior
    year=2023
)