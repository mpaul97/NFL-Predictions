import pandas as pd
import numpy as np
import os
import datetime
import time

class LastAvgN:
    def __init__(self, n: int, df: pd.DataFrame, _dir: str):
        self.n = n
        self.df = df
        self.df: pd.DataFrame = self.add_datetime_columns(self.df)
        self._dir = _dir
        self.drop_cols = [
            'attendance', 'stadium_id', 'lineHit', 
            'month', 'ouHit', 'time', 
            'surface', 'duration'
        ]
        self.cols = [
            'first_downs', 'fourth_down_attempts', 'fourth_down_conversions',
            'fumbles', 'fumbles_lost', 'interceptions',
            'net_pass_yards', 'pass_attempts', 'pass_completions',
            'pass_touchdowns', 'pass_yards', 'penalties',
            'points', 'rush_attempts', 'rush_touchdowns',
            'rush_yards', 'third_down_attempts', 'third_down_conversions',
            'time_of_possession', 'times_sacked', 'total_yards',
            'turnovers', 'yards_from_penalties', 'yards_lost_from_sacks'
        ]
        self.stat_cols = [(prefix + col) for prefix in ['home_', 'away_'] for col in self.cols]
        self.new_cols = [(prefix + col) for prefix in ['for_', 'opp_'] for col in self.cols]
        self.new_cols = [('last_avg_n_' + str(self.n) + '_' + prefix + col) for prefix in ['home_', 'away_'] for col in self.new_cols]
        return
    def get_datetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def add_datetime_columns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.get_datetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def rename_cols(self, df: pd.DataFrame, is_home: bool):
        col_names = list(df.columns)
        if is_home:
            for name in col_names:
                new_col_name = name.replace('home_', 'for_').replace('away_', 'opp_')
                df = df.rename(columns={name: new_col_name})
        else:
            for name in col_names:
                new_col_name = name.replace('away_', 'for_').replace('home_', 'opp_')
                df = df.rename(columns={name: new_col_name})
        return df
    def get_stats(self, abbr: str, dt: datetime.datetime):
        stats = self.df.loc[
            (self.df['datetime']<dt)&
            ((self.df['home_abbr']==abbr)|(self.df['away_abbr']==abbr)),
            ['home_abbr']+self.stat_cols
        ].tail(self.n)
        home_stats = stats.loc[stats['home_abbr']==abbr, self.stat_cols]
        away_stats = stats.loc[stats['home_abbr']!=abbr, self.stat_cols]
        all_stats = pd.concat([self.rename_cols(home_stats, True), self.rename_cols(away_stats, False)])
        return all_stats.mean().values
    def func(self, row: pd.Series):
        home_abbr, away_abbr, dt = row[['home_abbr', 'away_abbr', 'datetime']]
        home_stats = self.get_stats(home_abbr, dt)
        away_stats = self.get_stats(away_abbr, dt)
        return np.concatenate([home_stats, away_stats])
    def build(self, source: pd.DataFrame, isNew: bool = False):
        fn: str = f'last_avg_n_{self.n}'
        if (fn + '.csv') in os.listdir(self._dir) and not isNew:
            print(f'{fn}.csv already built.')
            return
        source: pd.DataFrame = self.add_datetime_columns(source)
        source[self.new_cols] = source.apply(lambda row: self.func(row), result_type='expand', axis=1)
        source.drop(columns=['week', 'year', 'datetime'], inplace=True)
        source.fillna(source.mean(), inplace=True)
        fn: str = fn if not isNew else (fn + '_new')
        source.to_csv("%s.csv" % (self._dir + fn), index=False)
        return

########################

# start = time.time()

# lan = LastAvgN(
#     5,
#     pd.read_csv("%s.csv" % "../../../data/oldGameData_94"),
#     "data/"
# )

# lan.build(pd.read_csv("%s.csv" % "data/source_new"), True)

# end = time.time()
# elapsed = end - start

# print(f"Time Elapsed: {elapsed}")