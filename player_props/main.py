import pandas as pd
import numpy as np
import os
from functools import reduce
import datetime
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

pd.options.mode.chained_assignment = None

class Main:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.raw_dir = self.data_dir + "raw/"
        self.targets_dir = self.data_dir + "targets/"
        self.features_dir = self.data_dir + "features/"
        self.position_dir = self._dir + "../data/positionData/"
        # frames
        self.target: pd.DataFrame = None
        self.train: pd.DataFrame = None
        self.cd: pd.DataFrame = None
        # info
        self.merge_cols = ['key', 'wy', 'p_id']
        self.positions = ['qb', 'rb', 'wr']
        self.feature_funcs = [
            self.avgStatsN_features
        ]
        self.stat_cols = {
            'qb': [
                'completed_passes','attempted_passes','passing_yards',
                'passing_touchdowns','interceptions_thrown','times_sacked',
                'yards_lost_from_sacks','longest_pass','quarterback_rating',
                'rush_attempts','rush_yards','rush_touchdowns','longest_rush',
                'completion_percentage','td_percentage','interception_percentage',
                'yards_per_attempt','adjusted_yards_per_attempt','yards_per_completion',
                'sack_percentage','net_gained_per_pass_attempt',
                'adjusted_net_yards_per_pass_attempt','rush_yards_per_attempt','volume_percentage'
            ],
            'rb': [
                'rush_attempts','rush_yards','rush_touchdowns',
                'longest_rush','times_pass_target','receptions',
                'receiving_yards','receiving_touchdowns','longest_reception',
                'fumbles','fumbles_lost','rush_yards_per_attempt',
                'receiving_yards_per_reception','catch_percentage',
                'receiving_yards_per_target','total_touches','yards_from_scrimmage',
                'scrimmage_yards_per_touch','total_touchdowns','touchdown_per_touch',
                'touchdown_per_reception','touchdown_per_rush','touchdown_per_target',
                'volume_percentage'
            ],
            'wr': [
                'rush_attempts','rush_yards','rush_touchdowns',
                'longest_rush','times_pass_target','receptions',
                'receiving_yards','receiving_touchdowns','longest_reception',
                'fumbles','fumbles_lost','rush_yards_per_attempt',
                'receiving_yards_per_reception','catch_percentage',
                'receiving_yards_per_target','total_touches','yards_from_scrimmage',
                'scrimmage_yards_per_touch','total_touchdowns','touchdown_per_touch',
                'touchdown_per_reception','touchdown_per_rush','touchdown_per_target',
                'volume_percentage'
            ]
        }
        return
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def build_targets(self):
        for pos in self.positions:
            df = pd.read_csv("%s.csv" % (self.data_dir + pos + "_data"))
            markets = list(set(df['market']))
            markets.sort()
            new_df = df[self.merge_cols].drop_duplicates()
            for m in markets:
                new_df[m] = np.nan
            for index, row in df.iterrows():
                key, wy, pid, market, line = row[self.merge_cols+['market', 'line']]
                idx = new_df.loc[(new_df['key']==key)&(new_df['wy']==wy)&(new_df['p_id']==pid)].index.values[0]
                if market in markets:
                    new_df.at[idx, market] = line
            # drop columns when percent nan greater than 10%
            for m in markets:
                nan_count = new_df[m].isna().sum()
                pct_nan = nan_count/df.shape[0]
                if pct_nan > 0.1:
                    new_df.drop(columns=[m], inplace=True)
            self.save_frame(new_df, (self.targets_dir + pos + "_targets"))
        return
    # features
    def avgStatsN_features(self, position: str, source: pd.DataFrame, isTest: bool):
        N = 5
        fn = position + "_avgStatsN_" + str(N)
        if (fn + ".csv") in os.listdir(self.features_dir) and not isTest:
            print(fn + " already created.")
            return
        df = self.cd
        cols = [('avgStatsN_' + str(N) + "_" + col) for col in self.stat_cols[position]]
        new_df = pd.DataFrame(columns=self.merge_cols+cols)
        for index, row in source.iterrows():
            self.print_progress_bar(index, source.shape[0], fn)
            pid, dt = row[['p_id', 'datetime']]
            stats: pd.DataFrame = df.loc[(df['p_id']==pid)&(df['datetime']<dt)].tail(N)
            if not stats.empty:
                stats = stats.mean(numeric_only=True).to_frame().transpose()
                stats = stats.drop(columns=['week', 'year'])
                stats = list(stats.values[0])
            else:
                stats = list(np.zeros(len(self.stat_cols[position])))
            new_df.loc[len(new_df.index)] = list(row[self.merge_cols].values) + stats
        self.save_frame(new_df, (self.features_dir + fn))
        return
    # end features
    def join(self, position: str, source: pd.DataFrame):
        source = source.drop(columns=['week', 'year', 'datetime'])
        for fn in os.listdir(self.features_dir):
            df = pd.read_csv(self.features_dir + fn)
            source = source.merge(df, on=self.merge_cols)
        self.save_frame(source, (self.data_dir + position + "_train"))
        return
    def get_ols_drops(self, X, y, threshold):
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        pdf: pd.Series = results.pvalues.sort_values()
        ols = pdf.to_frame()
        ols.insert(0, 'name', ols.index)
        ols.columns = ['name', 'val']
        ols.fillna(1, inplace=True)
        drops = []
        for index, row in ols.iterrows():
            name = row['name']
            val = row['val']
            if val > threshold and name != 'const':
                drops.append(name)
        return drops
    def predict(self):
        target_cols = [col for col in self.target.columns if col not in self.merge_cols]
        for target_name in target_cols:
            df = self.target[self.merge_cols+[target_name]]
            df.dropna(inplace=True)
            df = df.merge(self.train, on=self.merge_cols)
            X = df.drop(columns=self.merge_cols+[target_name])
            y = df[target_name]
            ols_drops = self.get_ols_drops(X, y, 0.2)
            X = X.drop(columns=ols_drops)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"{target_name} - Accuracy: {acc}")
        return
    def build(self):
        position = 'qb'
        self.set_target(position)
        source: pd.DataFrame = self.target[self.merge_cols]
        source = self.addDatetimeColumns(source)
        self.set_cd(position)
        self.cd = self.addDatetimeColumns(self.cd)
        [func(position, source, False) for func in self.feature_funcs]
        self.join(position, source)
        self.set_train(position)
        self.predict()
        return
    def set_train(self, position: str):
        self.train = pd.read_csv("%s.csv" % (self.data_dir + position + "_train"))
        return
    def set_cd(self, position: str):
        self.cd = pd.read_csv("%s.csv" % (self.position_dir + position.upper() + "Data"))
        self.cd.drop(columns=['isHome'], inplace=True)
        return
    def set_target(self, position: str):
        self.target = pd.read_csv("%s.csv" % (self.targets_dir + position + "_targets"))
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

######################

m = Main("./")

# m.build_targets()
m.build()