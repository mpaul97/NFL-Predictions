import pandas as pd
import numpy as np
import os
import datetime
import random
import regex as re
import pickle
from functools import reduce

from GradeGui import GradeGui

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

class Main:
    def __init__(self, _dir: str):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.models_dir = self._dir + "models/"
        self.game_dir = self._dir + "../data/"
        self.position_dir = self._dir + "../data/positionData/"
        self.snap_dir = self._dir + "../snapCounts/"
        self.starters_dir = self._dir + "../starters/"
        # params
        self.positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        self.off_positions = ['QB', 'RB', 'WR', 'TE', 'OL']
        self.target_stats = {
            'QB': [
                'passing_yards', 'passing_touchdowns', 'interceptions_thrown',
                'quarterback_rating', 'rush_touchdowns', 'completion_percentage',
                'times_sacked'
            ],
            'RB': [
                'yards_from_scrimmage', 'total_touchdowns', 'rush_yards_per_attempt',
                'rush_yards', 'receiving_yards', 'total_touches'
            ],
            'WR': [
                'catch_percentage', 'receiving_yards',
                'receiving_touchdowns', 'receiving_yards_per_target'
            ],
            'TE': [
                'catch_percentage', 'receiving_yards',
                'receiving_touchdowns', 'receiving_yards_per_target'
            ],
            'OL': [
                'times_sacked', 'yards_lost_from_sacks', 'sack_percentage', 
                'rush_yards_per_attempt'
            ],
            'DL': [
                'sacks', 'combined_tackles', 'tackles_for_loss',
                'quarterback_hits'
            ],
            'LB': [
                'sacks', 'combined_tackles', 'tackles_for_loss',
                'quarterback_hits', 'interceptions'
            ],
            'DB': [
                'interceptions', 'passes_defended', 'combined_tackles',
                'solo_tackles'
            ]
        }
        self.agg_stats = {
            'QB': {
                'passing_yards': 'mean', 'passing_touchdowns': 'sum', 'interceptions_thrown': 'sum',
                'quarterback_rating': 'mean', 'rush_touchdowns': 'sum', 'completion_percentage': 'mean',
                'times_sacked': 'sum', 'isStarter': 'sum', 'off_pct': 'mean'
            },
            'RB': {
                'rush_yards': 'mean', 'yards_from_scrimmage': 'mean', 'total_touchdowns': 'sum',
                'rush_yards_per_attempt': 'mean', 'receiving_yards': 'mean', 'total_touches': 'mean',
                'isStarter': 'sum', 'off_pct': 'mean'
            },
            'WR': {
                'catch_percentage': 'mean', 'receiving_yards': 'mean',
                'receiving_touchdowns': 'sum', 'receiving_yards_per_target': 'mean',
                'isStarter': 'sum', 'off_pct': 'mean'
            },
            'TE': {
                'catch_percentage': 'mean', 'receiving_yards': 'mean',
                'receiving_touchdowns': 'sum', 'receiving_yards_per_target': 'mean',
                'isStarter': 'sum', 'off_pct': 'mean'
            },
            'OL': {
                'times_sacked': 'mean', 'yards_lost_from_sacks': 'mean', 'sack_percentage': 'mean', 
                'rush_yards_per_attempt': 'mean', 'isStarter': 'sum', 'off_pct': 'mean'
            },
            'DL': {
                'sacks': 'mean', 'combined_tackles': 'mean', 'tackles_for_loss': 'mean',
                'quarterback_hits': 'mean', 'isStarter': 'sum', 'def_pct': 'mean'
            },
            'LB': {
                'sacks': 'mean', 'combined_tackles': 'mean', 'tackles_for_loss': 'mean',
                'quarterback_hits': 'mean', 'interceptions': 'sum', 'isStarter': 'sum', 
                'def_pct': 'mean'
            },
            'DB': {
                'interceptions': 'sum', 'passes_defended': 'mean', 'combined_tackles': 'mean',
                'solo_tackles': 'mean', 'isStarter': 'sum', 'def_pct': 'mean'
            }
        }
        self.merge_cols = ['p_id', 'key', 'abbr', 'wy']
        # frames
        self.df: pd.DataFrame = None
        self.train: pd.DataFrame = None
        self.all_data: pd.DataFrame = None
        self.grades: pd.DataFrame = None
        self.cd = pd.read_csv("%s.csv" % (self.game_dir + "gameData"))
        self.snaps = pd.read_csv("%s.csv" % (self.snap_dir + "snap_counts"))
        self.sdf = pd.read_csv("%s.csv" % (self.starters_dir + "allStarters"))
        # models
        self.scaler: StandardScaler = None
        self.model: LogisticRegression = None
        return
    def zero_division(self, num, dem):
        try:
            return num/dem
        except ZeroDivisionError:
            return 0
        return
    def get_datetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def add_datetime_columns(self, df: pd.DataFrame):
        df['week'] = df['wy'].apply(lambda x: int(x.split(" | ")[0]))
        df['year'] = df['wy'].apply(lambda x: int(x.split(" | ")[1]))
        df['datetime'] = [self.get_datetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def get_is_starter(self, row: pd.Series):
        starters = self.sdf.loc[(self.sdf['key']==row['key'])&(self.sdf['abbr']==row['abbr']), 'starters'].values[0]
        return 1 if row['p_id'] in starters else 0
    def get_career_starts(self, row: pd.Series):
        all_starts = '|'.join(self.sdf.loc[self.sdf['datetime']<row['datetime'], 'starters'].values)
        return all_starts.count(row['p_id'])
    def get_per_game_data(self, position: str, updating: bool):
        self.set_df(position)
        df = self.df.copy()
        start = df.loc[df['wy'].str.contains('2012')].index.values[0]
        df: pd.DataFrame = df.loc[df.index>=start]
        if updating:
            df = df.loc[df['wy'].str.contains('2023')]
        df = df.reset_index(drop=True)
        df = df[self.merge_cols+self.target_stats[position]]
        pct_col = 'off_pct' if position in self.off_positions else 'def_pct'
        snaps = self.snaps[self.merge_cols+[pct_col]]
        df = df.merge(snaps, on=self.merge_cols)
        df['isStarter'] = df.apply(lambda x: self.get_is_starter(x), axis=1)
        return df
    def build_all_train(self, position: str, updating: bool):
        """
        Build/collect ALL training data for position
        Args:
            position (str): train position
        """
        df = self.get_per_game_data(position, updating)
        df_list = []
        wys = list(set(df['wy']))
        for wy in wys:
            week, year = [int(w) for w in wy.split(" | ")]
            season = (year-1) if week == 1 else year
            start = df.loc[df['wy']==wy].index.values[0]
            data: pd.DataFrame = df.loc[(df.index<start)&(df['wy'].str.contains(str(season)))]
            stats = data.groupby(['p_id']).agg(self.agg_stats[position])
            stats['wy'] = wy
            stats.sort_values(by=[self.target_stats[position][0]], ascending=False, inplace=True)
            df_list.append(stats)
        new_df = pd.concat(df_list)
        new_df = new_df.merge(df[self.merge_cols], on=['p_id', 'wy'])
        new_cols = self.merge_cols + [col for col in new_df.columns if col not in self.merge_cols]
        new_df = new_df[new_cols]
        # add end stats
        last_wy = df['wy'].values[-1]
        last_week, last_year = [int(w) for w in (last_wy).split(" | ")]
        data: pd.DataFrame = df.loc[df['wy'].str.contains(str(last_year))]
        stats = data.groupby(['p_id']).agg(self.agg_stats[position])
        stats['wy'] = str(last_week+1) + " | " + str(last_year)
        stats.reset_index(inplace=True)
        stats.sort_values(by=[self.target_stats[position][0]], ascending=False, inplace=True)
        new_df = pd.concat([new_df, stats])
        # end add stats
        snap_col = 'off_pct' if position in self.off_positions else 'def_pct'
        new_df.rename(columns={'isStarter': 'starts', snap_col: 'snap_pct'}, inplace=True)
        # career starts for OL
        if position == 'OL':
            new_df = self.add_datetime_columns(new_df)
            self.sdf = self.add_datetime_columns(self.sdf)
            new_df['career_starts'] = new_df.apply(lambda x: self.get_career_starts(x), axis=1)
            new_df.drop(columns=['week', 'year', 'datetime'], inplace=True)
        new_df = new_df.round(2)
        new_df.sort_values(by=['key'], inplace=True)
        if not updating:
            self.save_frame(new_df, (self.data_dir + position + "_all"))
        return new_df
    def build_sample(self, position: str):
        """
        Build/collect training data for position and display
        GUI to get user inputted grades
        Args:
            position (str): train position
        """
        self.set_all_data(position)
        df = self.all_data.sample(n=50, random_state=random.randrange(1, 42))
        gg = GradeGui(df, self.merge_cols)
        df['grade'] = gg.radio_values
        # filenaming
        fn = position + "_data_v"
        files = ','.join(os.listdir(self.data_dir))
        iteration = str(1) if fn not in files else str(max([int(f.replace(fn, '').replace('.csv', '')) for f in os.listdir(self.data_dir) if fn in f])+1)
        fn += iteration
        print(f"{position} - iteration: {iteration}")
        self.save_frame(df, (self.data_dir + fn))
        return
    def save_model(self, position: str):
        self.set_train(position)
        data = self.train
        X = data.drop(columns=self.merge_cols+['grade'])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pickle.dump(scaler, open(self.models_dir + position + "_scaler.sav", "wb"))
        y = data['grade']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression() if position not in ['OL'] else LinearRegression()
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"Accuracy: {acc}")
        pickle.dump(model, open(self.models_dir + position + "_grade.sav", "wb"))
        return
    def predict_all(self, position: str):
        self.set_all_data(position)
        self.set_scaler(position)
        self.set_model(position)
        X = self.all_data.drop(columns=self.merge_cols)
        X = self.scaler.transform(X)
        preds = self.model.predict(X)
        df = self.all_data[self.merge_cols]
        df['grade'] = preds
        self.set_df(position)
        new_df = self.df.loc[self.df['wy']=='1 | 2012', self.merge_cols]
        new_df['grade'] = [random.uniform(1.0, 3.0) for _ in range(len(new_df.index))]
        df = pd.concat([new_df, df])
        df = df.round(2)
        self.save_frame(df, (self.data_dir + position + "_grades"))
        return
    def build_all_grades(self):
        for position in self.positions:
            self.predict_all(position)
        return
    def test_grades(self, position: str):
        self.set_grades(position)
        df = self.grades.loc[self.grades['wy']=='11 | 2023']
        df.sort_values(by=['grade'], ascending=False, inplace=True)
        print(df['p_id'].values)
        # means = df.groupby(['p_id']).mean()
        # means.sort_values(by=['grade'], ascending=False, inplace=True)
        # for index, row in means.iterrows():
        #     print(index, row['grade'])
        return
    def build_ranks(self):
        grades = { fn[:2]: pd.read_csv(self.data_dir + fn) for fn in os.listdir(self.data_dir) if '_grades' in fn }
        df_list = []
        for pos in grades:
            rank_col = pos.lower() + 's'
            new_df = pd.DataFrame(columns=['wy', 'abbr', rank_col])
            df: pd.DataFrame = grades[pos]
            wys = list(set(df['wy']))
            for index, wy in enumerate(wys):
                self.print_progress_bar(index, len(wys), (pos + " - playerRanks"))
                year = wy.split(" | ")[1]
                s0 = df.loc[df['wy']==wy].index.values[0]
                data: pd.DataFrame = df.loc[(df.index<s0)&(df['wy'].str.contains(year))]
                total_weeks = len(set(data['wy']))
                try:
                    s1 = self.sdf.loc[self.sdf['wy']==wy].index.values[0]
                    starters = '|'.join(self.sdf.loc[(self.sdf.index<s1)&(self.sdf['wy'].str.contains(year)), 'starters'].values)
                except IndexError: # current week, no starters -> use all
                    starters = '|'.join(self.sdf.loc[self.sdf['wy'].str.contains(year), 'starters'].values)
                means = data.groupby(['p_id', 'abbr']).mean()
                means.reset_index(inplace=True)
                means['grade'] = means.apply(lambda x: x['grade'] + ((starters.count(x['p_id'])/total_weeks)*3), axis=1)
                means = means.round(2)
                abbrs = list(set(means['abbr']))
                for abbr in abbrs:
                    info = means.loc[means['abbr']==abbr, ['p_id', 'grade']].sort_values(by=['grade'], ascending=False)
                    pids = '|'.join(info['p_id'].values)
                    new_df.loc[len(new_df.index)] = [wy, abbr, pids]
            df_list.append(new_df)
        rank_df = reduce(lambda x, y: pd.merge(x, y, on=['wy', 'abbr']), df_list)
        rank_df = self.add_datetime_columns(rank_df)
        rank_df.sort_values(by=['datetime'], inplace=True)
        rank_df.drop(columns=['week', 'year', 'datetime'], inplace=True)
        rank_df = rank_df[['wy', 'abbr']+[(pos.lower() + 's') for pos in self.positions]]
        self.save_frame(rank_df, "playerRanks")
        return
    def update(self):
        """
        Update all_data, grades, and playerRanks
        """
        self.positions = ['QB']
        for position in self.positions:
            self.set_all_data(position)
            # shorten cd -> 1 | 2012 no data before
            start = self.cd.loc[self.cd['wy']=='2 | 2012'].index.values[0]
            cd: pd.DataFrame = self.cd.loc[self.cd.index>=start].reset_index(drop=True)
            missing_wys = list(set(cd['wy']).difference(self.all_data['wy']))
            missing_wys = ['5 | 2023']
            df = self.build_all_train(position, True)
            print(df)
        return
    def set_grades(self, position: str):
        self.grades = pd.read_csv("%s.csv" % (self.data_dir + position + "_grades"))
        return
    def set_model(self, position: str):
        self.model = pickle.load(open(self.models_dir + position + "_grade.sav", "rb"))
        return
    def set_scaler(self, position: str):
        self.scaler = pickle.load(open(self.models_dir + position + "_scaler.sav", "rb"))
        return
    def set_train(self, position: str):
        self.train = pd.concat([pd.read_csv(self.data_dir + fn) for fn in os.listdir(self.data_dir) if (position + '_data') in fn])
        return
    def set_all_data(self, position: str):
        self.all_data = pd.read_csv("%s.csv" % (self.data_dir + position + "_all"))
        return
    def set_df(self, position: str):
        if position != 'OL':
            self.df = pd.read_csv("%s.csv" % (self.position_dir + position + "Data")) if position not in ['DL', 'LB'] else pd.read_csv("%s.csv" % (self.position_dir + "LBDLData"))
            if position in ['DL', 'LB']:
                self.df = self.df.loc[self.df['position']==position].reset_index(drop=True)
        else:
            self.df = pd.read_csv("%s.csv" % (self.position_dir + "OLStatsData/OLStatsData"))
        self.df.rename(columns={'game_key': 'key'}, inplace=True)
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
    _dir="./"
)

# position = 'DB'

# m.build_all_train(position)

# m.build_sample(position)

# m.save_model(position)
# m.predict_all(position)

# m.build_all_grades()

# m.build_ranks()

# m.update()