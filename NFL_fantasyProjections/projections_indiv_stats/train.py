import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

pd.options.mode.chained_assignment = None

class Train:
    def __init__(self, _dir):
        self.positions = ['QB', 'RB', 'WR', 'TE']
        self.merge_cols = ['p_id', 'year', 'position']
        self._dir = _dir
        self.data_dir = _dir + 'data/'
        self.old_dir = _dir + '../projections/data/'
        self.fp_dir = _dir + "../data/"
        # frames
        self.fp: pd.DataFrame = pd.read_csv("%s.csv" % (_dir + "../../data/fantasyData_expanded"))
        self.starters_23: pd.DataFrame = pd.read_csv("%s.csv" % (_dir + "../../data/starters_23/starters_w1"))
        self.cd_frames = { pos: pd.read_csv("%s.csv" % (_dir + "../../data/positionData/" + pos + "Data")) for pos in self.positions }
        self.proj_frames = { pos: pd.read_csv("%s.csv" % (_dir + "../../data/allVolumeProjections/projectionData/" + pos + "_volume_projections")) for pos in self.positions }
        self.source_frames = { pos: None for pos in self.positions }
        self.test_source_frames = { pos: None for pos in self.positions }
        self.total_frames = { pos: None for pos in self.positions }
        self.target_frames = { pos: None for pos in self.positions }
        self.train_frames = { pos: None for pos in self.positions }
        self.test_frames = { pos: None for pos in self.positions }
        # other
        self.position_dirs = { pos: (self.data_dir + pos + '/') for pos in self.positions }
        self.feature_dirs = { pos: (self.data_dir + pos + '/features/') for pos in self.positions }
        self.test_dirs = { pos: (self.data_dir + pos + '/test/') for pos in self.positions }
        self.model_dirs = { pos: (self.data_dir + pos + '/models/') for pos in self.positions }
        self.target_cols = {
            'QB': ['passing_yards', 'passing_touchdowns', 'rush_yards', 'rush_touchdowns', 'interceptions_thrown'],
            'RB': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
            'WR': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
            'TE': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
        }
        self.extra_target_cols = {
            'QB': ['over_300_passing_yards', 'over_100_rush_yards'],
            'RB': ['over_100_rush_yards', 'over_100_receiving_yards'],
            'WR': ['over_100_rush_yards', 'over_100_receiving_yards'],
            'TE': ['over_100_rush_yards', 'over_100_receiving_yards']
        }
        self.feature_funcs = [
            self.careerAvg_feature, self.isStarter_feature, self.perGameAvg_feature,
            self.yearsInLeague_feature, self.lastSeasonStats_feature, self.isNewStarter_feature
        ]
        # stores tuples of target name + model
        self.models = { pos: {} for pos in self.positions }
        return
    def oldTargetToSource(self):
        """
        Creates new target for each position.
        """
        df = pd.read_csv("%s.csv" % (self.old_dir + "target"))
        all_pids = list(set(df['p_id'].values))
        new_df = pd.DataFrame(columns=['p_id', 'position'])
        for index, pid in enumerate(all_pids):
            self.printProgressBar(index, len(all_pids), 'oldTargetToSource')
            data: pd.Series = self.fp.loc[self.fp['p_id']==pid, 'position'].value_counts()
            data.sort_values(ascending=False, inplace=True)
            position = data.index[0]
            new_df.loc[len(new_df.index)] = [pid, position]
        df = df.merge(new_df, on=['p_id'])
        df.drop(columns=['points'], inplace=True)
        for pos in self.positions:
            temp_df = df.loc[df['position']==pos]
            temp_df.sort_values(by=['year'], inplace=True)
            self.saveFrame(temp_df, (self.position_dirs[pos] + 'source'))
        return
    def createTotals(self):
        for pos in self.positions:
            print(f"{pos} target")
            cols = self.target_cols[pos]
            extra_cols = self.extra_target_cols[pos]
            df: pd.DataFrame = self.cd_frames[pos]
            df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
            years = list(set(df['year']))
            years.sort()
            new_df = pd.DataFrame(columns=['p_id', 'year']+cols+extra_cols)
            info = { year: list(set(df.loc[df['year']==year, 'p_id'].values)) for year in years }
            for year in info:
                print(year)
                pids = info[year]
                for pid in pids:
                    stats = df.loc[(df['p_id']==pid)&(df['year']==year), cols].values
                    totals = np.sum(stats, axis=0)
                    # extra stats - # games over threshold
                    if pos == 'QB':
                        o_pass_yards = len(df.loc[(df['p_id']==pid)&(df['year']==year)&(df['passing_yards']>=300), 'p_id'].values)
                        o_rush_yards = len(df.loc[(df['p_id']==pid)&(df['year']==year)&(df['rush_yards']>=100), 'p_id'].values)
                        extra_totals = [o_pass_yards, o_rush_yards]
                    else:
                        o_rush_yards = len(df.loc[(df['p_id']==pid)&(df['year']==year)&(df['rush_yards']>=100), 'p_id'].values)
                        o_receiving_yards = len(df.loc[(df['p_id']==pid)&(df['year']==year)&(df['receiving_yards']>=100), 'p_id'].values)
                        extra_totals = [o_rush_yards, o_receiving_yards]
                    new_df.loc[len(new_df.index)] = [pid, year] + list(totals) + extra_totals
            _dir = self.position_dirs[pos]
            self.saveFrame(new_df, (_dir + "all_totals"))
        return
    def createTargets(self):
        self.setSourceFrames()
        self.setTotalFrames()
        for pos in self.position_dirs:
            _dir = self.position_dirs[pos]
            source: pd.DataFrame = self.source_frames[pos]
            totals: pd.DataFrame = self.total_frames[pos]
            df = source.merge(totals, on=['p_id', 'year'])
            self.saveFrame(df, (_dir + 'target'))
        return
    def careerAvg_feature(self, pos: str, source: pd.DataFrame, _dir):
        """
        Gets career averages for each target column of prior seasons. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        """
        name = 'careerAvg'
        if (name + '.csv') in os.listdir(_dir):
            print(pos + "-" + name + " already created.")
            return
        df: pd.DataFrame = self.total_frames[pos]
        cols = self.target_cols[pos] + self.extra_target_cols[pos]
        new_df = pd.DataFrame(columns=self.merge_cols+[(col + '_' + name) for col in cols])
        for index, (pid, year) in enumerate(source[['p_id', 'year']].values):
            self.printProgressBar(index, len(source.index), (pos + '_' + name))
            stats = df.loc[(df['p_id']==pid)&(df['year']<year), cols].values
            if len(stats) != 0:
                means = np.mean(stats, axis=0)
            else:
                means = np.zeros(len(cols))
            new_df.loc[len(new_df.index)] = [pid, year, pos] + list(means)
        self.saveFrame(new_df, (_dir + name))
        return
    def isStarter_feature(self, pos: str, source: pd.DataFrame, _dir):
        """
        Gets if player is starter. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        """
        name = 'isStarter'
        if (name + '.csv') in os.listdir(_dir):
            print(pos + "-" + name + " already created.")
            return
        df: pd.DataFrame = self.cd_frames[pos]
        new_df = pd.DataFrame(columns=self.merge_cols+[name])
        total_mean = np.mean(df['volume_percentage'])
        all_starters = '|'.join(self.starters_23['starters'].values)
        for index, (pid, year) in enumerate(source[['p_id', 'year']].values):
            if year != 2023:
                self.printProgressBar(index, len(source.index), (pos + '_' + name))
                vols = df.loc[(df['p_id']==pid)&(df['wy'].str.contains(str(year))), 'volume_percentage'].values
                if len(vols) != 0:
                    vol_mean = np.mean(vols)
                    isStarter = 1 if ((vol_mean > total_mean) and (len(vols) > 10)) else 0
                else:
                    isStarter = 0
            else:
                isStarter = 1 if pid in all_starters else 0
            new_df.loc[len(new_df.index)] = [pid, year, pos, isStarter]
        self.saveFrame(new_df, (_dir + name))
        return
    def perGameAvg_feature(self, pos: str, source: pd.DataFrame, _dir):
        """
        Gets per game averages for each target column of prior seasons. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        """
        name = 'perGameAvg'
        if (name + '.csv') in os.listdir(_dir):
            print(pos + "-" + name + " already created.")
            return
        cd: pd.DataFrame = self.cd_frames[pos]
        cd['year'] = [int(wy.split(" | ")[1]) for wy in cd['wy'].values]
        pj: pd.DataFrame = self.proj_frames[pos]
        pj['year'] = [int(wy.split(" | ")[1]) for wy in pj['wy'].values]
        cols = self.target_cols[pos]
        new_df = pd.DataFrame(columns=self.merge_cols+[(col + '_' + name) for col in cols])
        for index, (pid, year) in enumerate(source[['p_id', 'year']].values):
            self.printProgressBar(index, len(source.index), (pos + '_' + name))
            stats = cd.loc[(cd['p_id']==pid)&(cd['year']<year)&(cd['volume_percentage']>0.95), cols]
            p_stats = pj.loc[(pj['p_id']==pid)&(pj['year']<year), cols]
            if len(stats.index) != 0 or len(p_stats.index) != 0:
                all_stats = pd.concat([stats, p_stats]).values
                means = np.mean(all_stats, axis=0)
            else:
                means = np.zeros(len(cols))
            new_df.loc[len(new_df.index)] = [pid, year, pos] + list(means)
        self.saveFrame(new_df, (_dir + name))
        return
    def yearsInLeague_feature(self, pos: str, source: pd.DataFrame, _dir):
        """
        Gets years in league prior to season. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        """
        name = 'yearsInLeague'
        if (name + '.csv') in os.listdir(_dir):
            print(pos + "-" + name + " already created.")
            return
        df: pd.DataFrame = self.total_frames[pos]
        new_df = pd.DataFrame(columns=self.merge_cols+[name])
        for index, (pid, year) in enumerate(source[['p_id', 'year']].values):
            self.printProgressBar(index, len(source.index), (pos + '-' + name))
            size = len(df.loc[(df['p_id']==pid)&(df['year']<year), 'p_id'].values)
            new_df.loc[len(new_df.index)] = [pid, year, pos, size]
        self.saveFrame(new_df, (_dir + name))
        return
    def lastSeasonStats_feature(self, pos: str, source: pd.DataFrame, _dir):
        """
        Gets prior season stats. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        """
        name = 'lastSeasonStats'
        if (name + '.csv') in os.listdir(_dir):
            print(pos + "-" + name + " already created.")
            return
        df: pd.DataFrame = self.total_frames[pos]
        cols = self.target_cols[pos] + self.extra_target_cols[pos]
        new_df = pd.DataFrame(columns=self.merge_cols+[(col + '_' + name) for col in cols])
        for index, (pid, year) in enumerate(source[['p_id', 'year']].values):
            self.printProgressBar(index, len(source.index), (pos + '-' + name))
            stats = df.loc[(df['p_id']==pid)&(df['year']==(year-1)), cols].values
            if len(stats) != 0:
                stats = stats[0]
            else:
                stats = np.zeros(len(cols))
            new_df.loc[len(new_df.index)] = [pid, year, pos] + list(stats)
        self.saveFrame(new_df, (_dir + name))
        return
    def isNewStarter_feature(self, pos: str, source: pd.DataFrame, _dir):
        """
        Gets if player is starter but did not start previous year. \n
        Args:
            df (pd.DataFrame): source(target)
            _dir (str): features (train_dir) or test_dir \n
        """
        name = 'isNewStarter'
        if (name + '.csv') in os.listdir(_dir):
            print(pos + "-" + name + " already created.")
            return
        df: pd.DataFrame = self.cd_frames[pos]
        new_df = pd.DataFrame(columns=self.merge_cols+[name])
        total_mean = np.mean(df['volume_percentage'])
        all_starters = '|'.join(self.starters_23['starters'].values)
        for index, (pid, year) in enumerate(source[['p_id', 'year']].values):
            self.printProgressBar(index, len(source.index), (pos + '_' + name))
            if year != 2023:
                vols = df.loc[(df['p_id']==pid)&(df['wy'].str.contains(str(year))), 'volume_percentage'].values
                if len(vols) != 0:
                    vol_mean = np.mean(vols)
                    isStarter = 1 if ((vol_mean > total_mean) and (len(vols) > 10)) else 0
                else:
                    isStarter = 0
            else:
                isStarter = 1 if pid in all_starters else 0
            prev_vols = df.loc[(df['p_id']==pid)&(df['wy'].str.contains(str(year-1))), 'volume_percentage'].values
            if len(prev_vols) != 0:
                prev_vol_mean = np.mean(prev_vols)
                prev_isStarter = 1 if ((prev_vol_mean > total_mean) and (len(prev_vols) > 10)) else 0
            else:
                prev_isStarter = 0
            isNewStarter = 1 if (isStarter == 1 and prev_isStarter == 0) else 0
            new_df.loc[len(new_df.index)] = [pid, year, pos, isNewStarter]
        self.saveFrame(new_df, (_dir + name))
        return
    def mergeFeatures(self, name: str, pos: str, source: pd.DataFrame, _dirs):
        for fn in os.listdir(_dirs[pos]):
            df = pd.read_csv(_dirs[pos] + fn)
            source = source.merge(df, on=self.merge_cols)
        self.saveFrame(source, (self.position_dirs[pos] + name))
        return
    def createTrains(self):
        self.setSourceFrames()
        self.setTotalFrames()
        for pos in self.feature_dirs:
            [func(pos, self.source_frames[pos], self.feature_dirs[pos]) for func in self.feature_funcs]
            self.mergeFeatures('train', pos, self.source_frames[pos], self.feature_dirs)
            print()
        return
    def saveModels(self):
        self.setTrains()
        self.setTargets()
        for pos in self.positions:
            df: pd.DataFrame = self.train_frames[pos]
            target: pd.DataFrame = self.target_frames[pos]
            cols = self.target_cols[pos] + self.extra_target_cols[pos]
            for t_col in cols:
                X = df.drop(columns=self.merge_cols)
                y = target[t_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LinearRegression(n_jobs=-1)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                print(f"target ({t_col}) accuracy: {acc}")
                self.writeModel(pos, model, t_col)
        return
    def heatmap(self, pos: str):
        self.setTrains()
        self.setTargets()
        train: pd.DataFrame = self.train_frames[pos]
        target_col = self.target_cols[pos][0]
        target: pd.DataFrame = self.target_frames[pos][self.merge_cols+[target_col]]
        data: pd.DataFrame = train.merge(target, on=self.merge_cols)
        data.drop(columns=self.merge_cols, inplace=True)
        corrmat = data.corr()
        k = 10
        cols = corrmat.nlargest(k, target_col)[target_col].index
        cm = np.corrcoef(data[cols].values.T)
        sns.set(font_scale=0.75)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        return
    def createTestSources(self):
        """
        Gets all ranks, splits by positions in test_source files. \n
        Returns nothing if all test_source files already exist.
        """
        founds = []
        for pos in self.position_dirs:
            founds.append('test_source.csv' in os.listdir(self.position_dirs[pos]))
        if not all(founds):
            df = pd.read_csv("%s.csv" % (self.fp_dir + 'all_ranks'))
            all_source = pd.concat(self.source_frames.values())
            df: pd.DataFrame = df.loc[df['p_id'].isin(all_source['p_id'])]
            new_df = pd.DataFrame(columns=['p_id', 'year', 'position'])
            for index, pid in enumerate(df['p_id'].values):
                self.printProgressBar(index, len(df.index), 'testSources')
                position = all_source.loc[all_source['p_id']==pid, 'position'].values[-1]
                new_df.loc[len(new_df.index)] = [pid, 2023, position]
            for pos in self.position_dirs:
                _dir = self.position_dirs[pos]
                temp_df = new_df.loc[new_df['position']==pos]
                self.saveFrame(temp_df, (_dir + 'test_source'))
            return
        print('All test_source.csv already built.')
        return
    def createTests(self):
        self.setSourceFrames()
        self.setTotalFrames()
        self.setModels()
        self.createTestSources()
        self.setTestSourceFrames()
        for pos in self.test_source_frames:
            [func(pos, self.test_source_frames[pos], self.test_dirs[pos]) for func in self.feature_funcs]
            self.mergeFeatures('test', pos, self.test_source_frames[pos], self.test_dirs)
            self.setTests()
            df: pd.DataFrame = self.test_frames[pos]
            cols = self.target_cols[pos] + self.extra_target_cols[pos]
            models = self.models[pos]
            source: pd.DataFrame = self.test_source_frames[pos]
            for t_col in cols:
                X = df.drop(columns=self.merge_cols)
                model: LinearRegression = models[t_col]
                preds = model.predict(X)
                source[t_col] = preds
            source.sort_values(by=cols[0], ascending=False, inplace=True)
            source = source.round(2)
            self.saveFrame(source, (self.position_dirs[pos] + 'predictions'))
            print(f"{pos} predictions created.\n")
        return
    def setTrains(self):
        for pos in self.position_dirs:
            self.train_frames[pos] = pd.read_csv("%s.csv" % (self.position_dirs[pos] + "train"))
        return
    def setTargets(self):
        for pos in self.position_dirs:
            self.target_frames[pos] = pd.read_csv("%s.csv" % (self.position_dirs[pos] + "target"))
        return
    def setTests(self):
        for pos in self.position_dirs:
            self.test_frames[pos] = pd.read_csv("%s.csv" % (self.position_dirs[pos] + "test"))
        return
    def setTotalFrames(self):
        for pos in self.position_dirs:
            self.total_frames[pos] = pd.read_csv("%s.csv" % (self.position_dirs[pos] + "all_totals"))
        return
    def setTestSourceFrames(self):
        for pos in self.position_dirs:
            self.test_source_frames[pos] = pd.read_csv("%s.csv" % (self.position_dirs[pos] + "test_source"))
        return
    def setSourceFrames(self):
        for pos in self.position_dirs:
            self.source_frames[pos] = pd.read_csv("%s.csv" % (self.position_dirs[pos] + "source"))
        return
    def writeModel(self, pos, model: LinearRegression, name: str):
        pickle.dump(model, open((self.model_dirs[pos] +  name + '.sav'), 'wb'))
        return
    def setModels(self):
        for pos in self.model_dirs:
            _dir = self.model_dirs[pos]
            for name in (self.target_cols[pos] + self.extra_target_cols[pos]):
                model: LinearRegression = pickle.load(open((_dir + name + '.sav'), 'rb'))
                self.models[pos][name] = model
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
    
# / END Train
    
#####################

t = Train('./')

# t.oldTargetToSource()

# t.createTotals()

# t.createTargets()

# t.createTrains()

# t.saveModels()

# t.heatmap('QB')

t.createTests()
