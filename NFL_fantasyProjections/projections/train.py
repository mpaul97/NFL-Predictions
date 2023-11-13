import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import time
from functools import reduce
import random

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

class Train:
    def __init__(self, _dir):
        self._dir = _dir
        self.merge_cols = ['p_id', 'year']
        self.df: pd.DataFrame = pd.read_csv("%s.csv" % (_dir + "../../data/fantasyData_expanded"))
        self.sl: pd.DataFrame = pd.read_csv("%s.csv" % (_dir + "../../data/seasonLength"))
        self.cd: pd.DataFrame = pd.read_csv("%s.csv" % (_dir + "../../data/gameData_regOnly"))
        self.starters: pd.DataFrame = pd.read_csv("%s.csv" % (_dir + "../../data/starters_23/starters_w1"))
        self.data_dir = _dir + "data/"
        self.info_dir = _dir + "data/info/"
        self.features_dir = _dir + "features/"
        self.test_dir = _dir + "test/"
        self.models_dir = _dir + "models/"
        self.fp_dir = _dir + "../data/"
        self.target: pd.DataFrame = None
        self.train: pd.DataFrame = None
        self.test: pd.DataFrame = None
        self.all_df: pd.DataFrame = None
        self.model: LinearRegression = None
        self.feature_funcs = [
            self.lastSeasonTotals_feature, self.careerAvg_feature, self.perGameAvg_feature,
            self.encodedPosition_feature, self.yearsInLeague_feature
        ]
        self.pos_encodings = {'QB': 0, 'RB': 1, 'TE': 2, 'WR': 3}
        self.team_stat_cols = [
            'net_pass_yards', 'rush_yards', 'pass_touchdowns',
            'rush_touchdowns', 'pass_attempts', 'rush_attempts',
            'points', 'total_yards'
        ]
        self.drops: list = []
        return
    def createTotals(self, isAll: bool):
        """
        Gets all season total points for each year. \n
        Args:
            isAll (bool): all data or just target data (>2000)
        """
        df = self.df
        # only data starting from 2000 to now
        if not isAll:
            start = df.loc[df['wy'].str.contains('2000')].index.values[0]
            df = df.loc[df.index>start]
        df: pd.DataFrame = df[['p_id', 'position', 'wy', 'points']]
        df['week'] = [int(wy.split(' | ')[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(' | ')[1]) for wy in df['wy'].values]
        # remove playoffs
        years = list(set(df['year']))
        years.sort()
        df_list = []
        for year in years:
            reg_weeks = self.sl.loc[self.sl['year']==year, 'weeks'].values[0]
            temp_df = df.loc[(df['year']==year)&(df['week']<=reg_weeks)]
            temp_df: pd.DataFrame = temp_df[['p_id', 'points']]
            totals = temp_df.groupby(['p_id']).sum()
            totals.insert(0, 'p_id', totals.index)
            totals.insert(1, 'year', year)
            totals.reset_index(drop=True, inplace=True)
            totals.sort_values(by=['points'], ascending=False, inplace=True)
            df_list.append(totals)
        new_df = pd.concat(df_list)
        new_df = new_df.loc[new_df['points']>2]
        fn = 'all_totals' if isAll else 'target'
        self.saveFrame(new_df, (self.data_dir + fn))
        return
    def lastSeasonTotals_feature(self, df: pd.DataFrame, _dir):
        """
        Gets last season totals. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        Returns:
            _type_: new_df
        """
        if 'lastSeasonTotals.csv' in os.listdir(_dir):
            print("lastSeasonTotals already created.")
            return
        df = df[self.merge_cols]
        new_df = pd.DataFrame(columns=['p_id', 'year', 'last_season_total'])
        for index, (pid, year) in enumerate(df[self.merge_cols].values):
            self.printProgressBar(index, len(df.index), 'lastSeasonTotals')
            try:
                last_total = self.all_df.loc[(self.all_df['p_id']==pid)&(self.all_df['year']==year-1), 'points'].values[0]
            except IndexError:
                last_total = 0
            new_df.loc[len(new_df.index)] = [pid, year, last_total]
        self.saveFrame(new_df, (_dir + 'lastSeasonTotals'))
        return new_df
    def careerAvg_feature(self, df: pd.DataFrame, _dir):
        """
        Gets career averages for prior seasons. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        Returns:
            _type_: new_df
        """
        if 'careerAvg.csv' in os.listdir(_dir):
            print("careerAvg already created.")
            return
        df = df[self.merge_cols]
        new_df = pd.DataFrame(columns=['p_id', 'year', 'career_avg'])
        for index, (pid, year) in enumerate(df[self.merge_cols].values):
            self.printProgressBar(index, len(df.index), 'careerAvg')
            stats = self.all_df.loc[(self.all_df['year']<year)&(self.all_df['p_id']==pid), 'points'].values
            mean = np.mean(stats) if len(stats) != 0 else 0
            new_df.loc[len(new_df.index)] = [pid, year, mean]
        self.saveFrame(new_df, (_dir + 'careerAvg'))
        return new_df
    def perGameAvg_feature(self, df: pd.DataFrame, _dir):
        """
        Gets per game averages for prior seasons. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        Returns:
            _type_: new_df
        """
        if 'perGameAvg.csv' in os.listdir(_dir):
            print("perGameAvg already created.")
            return
        cd = self.df
        df = df[self.merge_cols]
        new_df = pd.DataFrame(columns=['p_id', 'year', 'per_game_avg'])
        for index, (pid, year) in enumerate(df[self.merge_cols].values):
            self.printProgressBar(index, len(df.index), 'per_game_avg')
            # start = cd.loc[cd['wy'].str.contains(str(year))].index.values[0]
            stats = cd.loc[(cd['year']<year)&(cd['p_id']==pid), 'points'].values
            mean = np.mean(stats) if len(stats) != 0 else 0
            new_df.loc[len(new_df.index)] = [pid, year, mean]
        self.saveFrame(new_df, (_dir + 'perGameAvg'))
        return new_df
    def encodedPosition_feature(self, df: pd.DataFrame, _dir):
        """
        Gets encoded position for player. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        Returns:
            _type_: new_df
        """
        if 'encodedPosition.csv' in os.listdir(_dir):
            print("encodedPosition already created.")
            return
        cd = self.df
        df = df[self.merge_cols]
        cd = cd[['p_id', 'position']]
        cd.drop_duplicates(inplace=True)
        positions = []
        for index, pid in enumerate(df['p_id'].values):
            self.printProgressBar(index, len(df.index), 'encodedPosition')
            try:
                positions.append(self.pos_encodings[cd.loc[cd['p_id']==pid, 'position'].values[0]])
            except IndexError:
                # print(f"No encoded position value for: {pid}")
                positions.append(-1)
        df['encoded_position'] = positions
        self.saveFrame(df, (_dir + 'encodedPosition'))
        return df
    def playerAbbrs(self, df: pd.DataFrame):
        # player abbrs - pid, year, abbr (most common for year)
        if 'playerAbbrs.csv' not in os.listdir(self.info_dir):
            all_abbrs = []
            for index, (pid, year) in enumerate(df[self.merge_cols].values):
                self.printProgressBar(index, len(df.index), 'playerAbbrs')
                abbrs: pd.Series = self.df.loc[(self.df['p_id']==pid)&(self.df['wy'].str.contains(str(year))), 'abbr'].value_counts()
                abbrs.sort_values(ascending=False, inplace=True)
                abbr = abbrs.index[0]
                all_abbrs.append(abbr)
            df['abbr'] = all_abbrs
            self.saveFrame(df, (self.info_dir + 'playerAbbrs'))
            return df
        # update playerAbbrs - 2023 players
        old_df = pd.read_csv("%s.csv" % (self.info_dir + 'playerAbbrs'))
        if 2023 not in old_df['year'].values:
            print('Updating playerAbbrs with test pids...')
            for index, pid in enumerate(df['p_id'].values):
                try: # get abbr from 2023 week 1 starters
                    abbr = self.starters.loc[self.starters['starters'].str.contains(pid), 'abbr'].values[0]
                except IndexError:
                    try: # get most recent available abbr
                        abbr = old_df.loc[old_df['p_id']==pid, 'abbr'].values[-1]
                    except IndexError: # get random abbr
                        abbr = self.starters['abbr'].values[random.randrange(0, len(self.starters.index))]
                old_df.loc[len(old_df.index)] = [pid, 2023, abbr]
            self.saveFrame(old_df, (self.info_dir + 'playerAbbrs'))
        return old_df
    def getTeamStats_season(self, abbr: str, year: int):
        home_stats = self.cd.loc[(self.cd['home_abbr']==abbr)&(self.cd['year']==year)]
        away_stats = self.cd.loc[(self.cd['away_abbr']==abbr)&(self.cd['year']==year)]
        _for, against = [], []
        for col in self.team_stat_cols:
            for_home_data = np.mean(home_stats[['home_' + col]].values)
            for_away_data = np.mean(away_stats[['away_' + col]].values)
            for_data = np.mean([for_home_data, for_away_data])
            _for.append(for_data)
            against_home_data = np.mean(home_stats[['away_' + col]].values)
            against_away_data = np.mean(away_stats[['home_' + col]].values)
            against_data = np.mean([against_home_data, against_away_data])
            against.append(against_data)
        return _for, against
    def teamStats(self):
        """
        Write all abbr team stats (passing_yards, allowed_passing_yards, etc.)
        for each year from 1994 (gameData start wy) to now.
        """
        if 'teamStats.csv' in os.listdir(self.info_dir):
            print('INFO - teamStats already created.')
            return pd.read_csv("%s.csv" % (self.info_dir + "teamStats"))
        data = self.cd[['home_abbr', 'year']]
        data.columns = ['abbr', 'year']
        data.drop_duplicates(inplace=True)
        cols = [(prefix + col) for prefix in ['for_', 'against_'] for col in self.team_stat_cols]
        new_df = pd.DataFrame(columns=['abbr', 'year']+cols)
        for index, (abbr, year) in enumerate(data[['abbr', 'year']].values):
            self.printProgressBar(index, len(data.index), 'teamStats')
            for_stats, against_stats = self.getTeamStats_season(abbr, year)
            new_df.loc[len(new_df.index)] = [abbr, year] + for_stats + against_stats
        # add ranks
        print("\nAdding ranks...")
        df_list = []
        for year in list(set(new_df['year'].values)):
            temp_df: pd.DataFrame = new_df.loc[new_df['year']==year]
            for col in cols:
                data = temp_df[['abbr', col]].values
                data = data[data[:, 1].argsort()]
                data = data if 'against' in col else data[::-1]
                s_data = pd.DataFrame(data, columns=['abbr', 'values'])
                s_data[col + '_rank'] = s_data.index + 1
                s_data.drop(columns=['values'], inplace=True)
                temp_df = temp_df.merge(s_data, on=['abbr'])
            df_list.append(temp_df)
        new_df = pd.concat(df_list)
        new_df = new_df.round(2)
        self.saveFrame(new_df, (self.info_dir + 'teamStats'))
        return new_df
    def teamStatsRanks_feature_not_using(self, df: pd.DataFrame, _dir):
        """
        Gets team stats and ranks for previous season. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        Returns:
            _type_: new_df
        """
        if 'teamStatsRanks.csv' in os.listdir(_dir):
            print("teamStatsRanks already created.")
            return
        df = df[self.merge_cols]
        pdf: pd.DataFrame = self.playerAbbrs(df)
        df = df.merge(pdf, on=self.merge_cols)
        t_stats: pd.DataFrame = self.teamStats()
        cols = list(t_stats.columns[2:])
        new_df = pd.DataFrame(columns=self.merge_cols+cols)
        for index, (pid, year, abbr) in enumerate(df[['p_id', 'year', 'abbr']].values):
            self.printProgressBar(index, len(df.index), 'teamStatsRanks')
            try:
                data = t_stats.loc[(t_stats['abbr']==abbr)&(t_stats['year']==(year-1))].values[0]
                data = data[2:]
            except IndexError:
                data = [np.nan for _ in range(len(cols))]
            new_df.loc[len(new_df.index)] = [pid, year] + list(data)
        new_df.fillna(new_df.mean(), inplace=True)
        self.saveFrame(new_df, (_dir + 'teamStatsRanks'))
        return new_df
    def yearsInLeague_feature(self, df: pd.DataFrame, _dir):
        """
        Gets years in league prior to current season. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        Returns:
            _type_: new_df
        """
        if 'yearsInLeague.csv' in os.listdir(_dir):
            print("yearsInLeague already created.")
            return
        df = df[self.merge_cols]
        new_df = pd.DataFrame(columns=['p_id', 'year', 'years_in_league'])
        data = self.df[['p_id', 'year']]
        data.drop_duplicates(inplace=True)
        for index, (pid, year) in enumerate(df[self.merge_cols].values):
            self.printProgressBar(index, len(df.index), 'years_in_league')
            count = len(data.loc[(data['p_id']==pid)&(data['year']<year), 'p_id'].values)
            new_df.loc[len(new_df.index)] = [pid, year, count]
        self.saveFrame(new_df, (_dir + 'yearsInLeague'))
        return new_df
    def mergeFeatures(self, name: str, _dir):
        fns: list = os.listdir(_dir)
        first_fn = fns[0] # use first feature as merge source
        fns.pop(0)
        new_df = pd.read_csv(_dir + first_fn)
        for fn in fns:
            df = pd.read_csv(_dir + fn)
            new_df = new_df.merge(df, on=self.merge_cols)
        self.saveFrame(new_df, (self.data_dir + name))
        return
    def createTrain(self):
        self.setTarget()
        self.setAllDf()
        [func(self.target, self.features_dir) for func in self.feature_funcs]
        self.mergeFeatures('train', self.features_dir)
        return
    def heatmap(self):
        self.setTrain()
        self.setTarget()
        data: pd.DataFrame = self.train.merge(self.target, on=self.merge_cols)
        corrmat = data.corr()
        k = 20
        cols = corrmat.nlargest(k, 'points')['points'].index
        cm = np.corrcoef(data[cols].values.T)
        sns.set(font_scale=0.75)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        return
    def getCorrColDrops(self, data: pd.DataFrame):
        corrmat = data.corr()
        k = 10
        data = data.drop(columns=['p_id', 'year'])
        cols = list(corrmat.nlargest(k, 'points')['points'].index)
        drops = list(set(data.columns).difference(set(cols)))
        if 'points' in drops:
            drops.remove('points')
        self.writeDrops(drops, (self.data_dir + 'drops'))
        return drops
    def saveModels(self, showPreds: bool = False):
        self.setTrain()
        self.setTarget()
        data: pd.DataFrame = self.train.merge(self.target, on=self.merge_cols)
        drops = self.getCorrColDrops(data)
        X = data.drop(columns=self.merge_cols+drops+['points'])
        y = data['points']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression(n_jobs=-1)
        try:
            model.fit(X_train, y_train)
        except ValueError: # non-countinous model
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"Accuracy: {acc}")
        if showPreds:
            preds = model.predict(X_test)
            for i in range(10):
                print(f"Predicted: {preds[i]}, Expected: {y_test.iloc[i]}")
        self.writeModel(model, 'points')
        return
    def createTest(self):
        self.setAllDf()
        self.setModel()
        self.setDrops()
        fns = [fn for fn in os.listdir(self.fp_dir) if 'clean' in fn]
        fns = ['clean_ranks_std.csv']
        for fn in fns:
            df = pd.read_csv(self.fp_dir + fn)
            df = df[['p_id']]
            df['year'] = 2023
            df = df.loc[df['p_id']!='UNK']
            [func(df, self.test_dir) for func in self.feature_funcs]
            self.mergeFeatures('test', self.test_dir)
            self.setTest()
            self.test = self.test.drop(columns=self.drops)
            preds = self.model.predict(self.test.drop(columns=self.merge_cols))
            self.test = self.test[['p_id', 'year']]
            self.test['pred_points'] = preds
            self.test.sort_values(by=['pred_points'], ascending=False, inplace=True)
            self.test.reset_index(drop=True, inplace=True)
            self.test['rank'] = self.test.index + 1
            self.test = self.test.round(2)
            self.saveFrame(self.test, (self.data_dir + 'predictions'))
        return
    def setTarget(self):
        self.target = pd.read_csv("%s.csv" % (self.data_dir + "target"))
        return
    def setTrain(self):
        self.train = pd.read_csv("%s.csv" % (self.data_dir + "train"))
        return
    def setTest(self):
        self.test = pd.read_csv("%s.csv" % (self.data_dir + "test"))
        return
    def setAllDf(self):
        self.all_df = pd.read_csv("%s.csv" % (self.data_dir + "all_totals"))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def writeDrops(self, drops: list, name: str):
        file = open((name + '.txt'), 'w')
        file.write(','.join(drops))
        file.close()
        return
    def setDrops(self):
        file = open((self.data_dir + 'drops.txt'), 'r')
        lines = file.readlines()
        try:
            self.drops = lines[0].split(",")
        except IndexError:
            print('No drops found.')
            self.drops = []
        return
    def writeModel(self, model: LinearRegression, name: str):
        pickle.dump(model, open((self.models_dir +  name + '.sav'), 'wb'))
        return
    def setModel(self):
        self.model: LinearRegression = pickle.load(open((self.models_dir + 'points.sav'), 'rb'))
        return
    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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
    
# / END Train

# Enumerate Time: 0.008997917175292969
# Iterrows Time: 0.5374598503112793

######################

t = Train('./')

# t.createTotals(isAll=False)

# t.createTrain()

# t.heatmap()

# t.saveModels(True)

# t.createTest()



