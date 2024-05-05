import pandas as pd
import numpy as np
import os
import datetime
import random

from collect import Collect
from processing import Processing

pd.options.mode.chained_assignment = None

class Main:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.features_dir = self._dir + "data/features/"
        self.game_data_dir = self._dir + "../data/"
        self.position_data_dir = self.game_data_dir + "positionData/"
        # frames
        self.df: pd.DataFrame = None
        self.gd = pd.read_csv("%s.csv" % (self.game_data_dir + "gameData"))
        # features
        self.feature_funcs = [
            self.possession_epas_feature, self.qb_career_epas_feature
        ]
        # info
        self.datetime_cols = ['week', 'year', 'datetime']
        return
    # helpers
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
    def get_datetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def add_datetime_columns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.get_datetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    # END helpers
    # getters/setters
    def get_df(self):
        """
        PBP total
        Returns:
            pd.DataFrame: merge DataFrame
        """
        df = Processing("./").get_total_pbp()
        df['location'] = df['location'].astype(str)
        df = df.loc[~df['location'].str.contains('Overtime')]
        df = self.gd[['key', 'wy']].merge(df, on=['key'])
        df: pd.DataFrame = self.add_datetime_columns(df)
        df['epa'] = df['epa'].astype(float)
        df['epb'] = df['epb'].astype(float)
        df['epa_added'] = df.apply(lambda row: row['epa']-row['epb'], axis=1)
        df = df.loc[~pd.isna(df['epa_added'])]
        return df
    # END getters/setters
    def update_all(self):
        """
        Write new tables, clean tables, update allTables
        Update/create names, possessions, and entities
        """
        collect = Collect("./")
        collect.updateTables()
        processing = Processing("./")
        processing.update()
        return
    def possession_epas_feature(self, df: pd.DataFrame):
        """
        Get offensive and defensive EPA added (EPA - EPB) + total EPA for each game
        Args:
            df (pd.DataFrame): total_pbp
        """
        fn = "possession_epas"
        if f'{fn}.csv' in os.listdir(self.features_dir):
            print(f"{fn} already created.")
            return
        edf = df.groupby(by=['key', 'possession']).mean()[['epa_added', 'epa']]
        edf = edf.reset_index()
        edf.columns = ['key', 'abbr', 'epa_added', 'epa']
        new_df = pd.DataFrame(columns=['key', 'home_abbr', 'away_abbr', 'home_epa_added', 'away_epa_added', 'home_total_epa', 'away_total_epa'])
        for i in range(0, len(edf.index), 2):
            a = edf.iloc[i]
            b = edf.iloc[i+1]
            abbrs = [a['abbr'], b['abbr']]
            home_abbr = (a['key'][-3:]).upper()
            abbrs.remove(home_abbr)
            home_dif = a['epa_added'] if home_abbr == a['abbr'] else b['epa_added']
            away_dif = b['epa_added'] if home_abbr == a['abbr'] else a['epa_added']
            home_epa = a['epa'] if home_abbr == a['abbr'] else b['epa']
            away_epa = b['epa'] if home_abbr == a['abbr'] else a['epa']
            new_df.loc[len(new_df.index)] = [a['key'], home_abbr, abbrs[0], home_dif, away_dif, home_epa, away_epa]
        new_df = new_df.merge(self.gd[['key', 'wy']], on=['key'])
        new_df = new_df[['key', 'wy', 'home_abbr', 'away_abbr', 'home_epa_added', 'away_epa_added', 'home_total_epa', 'away_total_epa']]
        self.save_frame(new_df, (self.features_dir + fn))
        return
    def qb_career_epas_feature(self, df: pd.DataFrame):
        """
        Get EPA added + total EPA when passer involved in play, ONLY either PASSER, RUSHER, PENALIZER, or FUMBLER
        Args:
            df (pd.DataFrame): total_pbp
        """
        cols = ['qb_career_epa_added', 'qb_career_total_epa']
        def get_qb_career_epas(row: pd.Series):
            print(cols)
            return { col: np.nan for col in cols }
        fn = "qb_epas"
        if f'{fn}.csv' in os.listdir(self.features_dir):
            print(f"{fn} already created.")
            return
        cd = pd.read_csv("%s.csv" % (self.position_data_dir + "QBData"))
        cd = cd[['p_id', 'game_key']]
        cd.columns = ['p_id', 'key']
        df = df.merge(cd, on=['key'])
        print(df.head())
        # cd = self.add_datetime_columns(cd)
        # cd = cd.loc[cd['datetime']>=self.get_datetime(1, 2011)]
        # cd.reset_index(drop=True, inplace=True)
        # cd = cd[['p_id', 'game_key']+self.datetime_cols]
        # cd = cd.tail(10)
        # cd[cols] = cd.apply(lambda x: get_qb_career_epas(x), axis=1, result_type='expand')
        return
    def build_features(self):
        df = self.get_df()
        [func(df) for func in self.feature_funcs]
        return
    
# END / Main

###################

# !!! Tables go from 2011 - 2023 championship games !!!

m = Main("./")

m.update_all()