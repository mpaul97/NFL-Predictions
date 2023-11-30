import pandas as pd
import numpy as np
import os
import pickle
import regex as re
import statsmodels.api as sm
from ordered_set import OrderedSet
import multiprocessing
from itertools import repeat
from functools import partial, reduce
import joblib
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import NuSVC, NuSVR, LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

pd.options.mode.chained_assignment = None

try:
    from fantasyPredictions.features.source.source import buildSource, buildNewSource
    from fantasyPredictions.features.seasonAvg.seasonAvg import buildSeasonAvg, buildNewSeasonAvg
    from fantasyPredictions.features.encodedPosition.encodedPosition import buildEncodedPosition, buildNewEncodedPosition
    from fantasyPredictions.features.maxWeekRank.maxWeekRank import buildMaxWeekRank, buildNewMaxWeekRank
    from fantasyPredictions.features.pointsN.pointsN import buildPointsN, buildNewPointsN
    from fantasyPredictions.features.careerAvg.careerAvg import buildCareerAvg, buildNewCareerAvg
    from fantasyPredictions.features.allowedPointsN.allowedPointsN import buildAllowedPointsN, buildNewAllowedPointsN
    from fantasyPredictions.features.seasonRankings.seasonRankings import buildSeasonRankings, buildNewSeasonRankings
    from fantasyPredictions.features.isPrimary.isPrimary import IsPrimary
    from fantasyPredictions.features.maddenRatings.maddenRatings import MaddenRatings
    from fantasyPredictions.features.isStarter.isStarter import IsStarter
    from fantasyPredictions.features.lastStatsN.lastStatsN import LastStatsN
    from fantasyPredictions.features.avgStatsN.avgStatsN import AvgStatsN
    from fantasyPredictions.features.advancedStats.advancedStats import AdvancedStats
    from fantasyPredictions.features.seasonAvgSnapPercentages.seasonAvgSnapPercentages import SeasonAvgSnapPercentages
    from fantasyPredictions.features.lastSnapPercentagesN.lastSnapPercentagesN import LastSnapPercentagesN
    from fantasyPredictions.features.lastSnapDifferencesN.lastSnapDifferencesN import LastSnapDifferencesN
except ModuleNotFoundError:
    print('No modules found.')
    
class Build:
    def __init__(self, all_paths: dict, _dir: str):
        self.all_paths = all_paths
        self._dir = _dir
        self.features_dir = _dir + 'features/'
        self.models_dir = _dir + 'models/'
        self.positions = ['QB', 'RB', 'WR', 'TE']
        # paths
        self.data_path = self.all_paths['dp']
        self.position_path = self.all_paths['pp']
        self.starters_path = self.all_paths['sp']
        self.team_names_path = self.all_paths['tnp']
        self.coaches_path = self.all_paths['cp']
        self.madden_path = self.all_paths['mrp']
        self.snap_path = self.all_paths['sc']
        # dataframes
        self.cd = pd.read_csv("%s.csv" % (self.data_path + "skillData"))
        self.fd = pd.read_csv("%s.csv" % (self.data_path + "fantasyData"))
        self.ocd = pd.concat([pd.read_csv("%s.csv" % (self.position_path + pos + "Data")) for pos in self.positions])
        self.pos_data = { pos: pd.read_csv("%s.csv" % (self.position_path + pos + "Data")) for pos in self.positions }
        self.gd = pd.read_csv("%s.csv" % (self.data_path + "gameData"))
        self.ogd = pd.read_csv("%s.csv" % (self.data_path + "oldGameData_78"))
        self.sl = pd.read_csv("%s.csv" % (self.data_path + "seasonLength"))
        self.rdf = pd.read_csv("%s.csv" % (self.madden_path + "playerRatings"))
        self.sdf = pd.read_csv("%s.csv" % (self.starters_path + "allStarters"))
        self.adf = pd.read_csv("%s.csv" % (self.data_path + "advancedStats"))
        self.scdf = pd.read_csv("%s.csv" % (self.snap_path + "snap_counts"))
        # more frames
        self.train: pd.DataFrame = None
        self.target: pd.DataFrame = None
        self.pred_train: pd.DataFrame = None
        self.pred_targets: pd.DataFrame = None
        self.trainInfo: pd.DataFrame = None
        self.drops_df: pd.DataFrame = None
        self.drops_positions_df: pd.DataFrame = None
        self.test: pd.DataFrame = None
        # models stuff
        self.str_cols = ['key', 'abbr', 'p_id', 'wy', 'position']
        self.non_cont_models = ['log', 'forest', 'knn']
        self.metric_models = ['knn']
        self.target_ols_thresholds = {
            'points': 0.2, 'week_rank': 0.2
        }
        self.all_models = {
            'points': {
                'linear': LinearRegression(n_jobs=-1)
            },
            'week_rank': {
                'forest': RandomForestClassifier(n_jobs=-1)
            }
        }
        self.position_all_models = {
            'points': {
                'forestReg': RandomForestRegressor(n_jobs=-1),
                'linear': LinearRegression(n_jobs=-1),
                'log': LogisticRegression(n_jobs=-1),
                # 'forest': RandomForestClassifier(n_jobs=-1)
            }
        }
        self.pred_target_stats = {
            'QB': ['passing_yards', 'passing_touchdowns', 'rush_yards', 'rush_touchdowns'],
            'RB': ['rush_yards', 'rush_touchdowns'],
            'WR': ['receiving_yards', 'receiving_touchdowns'],
            'TE': ['receiving_yards', 'receiving_touchdowns']
        }
        self.pred_target_models = {
            'QB': {
                'passing_yards': {
                    'forestReg': RandomForestRegressor(n_jobs=-1)
                },
                'passing_touchdowns': {
                    'lsvc': LinearSVC()
                },
                'rush_yards': {
                    'forestReg': RandomForestRegressor(n_jobs=-1)
                },
                'rush_touchdowns': {
                    'lsvc': LinearSVC()
                }
            },
            'RB': {
                'rush_yards': {
                    'forestReg': RandomForestRegressor(n_jobs=-1)
                },
                'rush_touchdowns': {
                    'lsvc': LinearSVC()
                }
            },
            'WR': {
                'receiving_yards': {
                    'forestReg': RandomForestRegressor(n_jobs=-1)
                },
                'receiving_touchdowns': {
                    'lsvc': LinearSVC()
                }
            },
            'TE': {
                'receiving_yards': {
                    'forestReg': RandomForestRegressor(n_jobs=-1)
                },
                'receiving_touchdowns': {
                    'lsvc': LinearSVC()
                }
            }
        }
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
    def saveFrame(self, df: pd.DataFrame, name: str):
        """
        Writes frame to self._dir with name(name).
        Args:
            df (pd.DataFrame): any frame
            name (str): filename
        """
        df.to_csv("%s.csv" % (self._dir + name), index=False)
        return
    def combineDir(self, _type):
        """
        Combines feature dir with feature type dir \n
        Args:
            _type (_type_): feature name/type
        Returns:
            _type_: feature root _dir + specific feature name
        """
        return self.features_dir + _type + '/'
    def getQbPoints(self, row: pd.Series):
        points = 0
        # passing_touchdowns
        points += round(row['passing_touchdowns'], 0)*4
        # passing_yards
        points += round(row['passing_yards'], 0)*0.04
        points += 3 if row['passing_yards'] > 300 else 0
        # interceptions
        points -= round(row['interceptions_thrown'], 0)
        # rush_yards
        points += round(row['rush_yards'], 0)*0.1
        points += 3 if row['rush_yards'] > 100 else 0
        # rush_touchdowns
        points += round(row['rush_touchdowns'], 0)*6
        return points
    def getSkillPoints(self, row: pd.Series):
        points = 0
        # rush_yards
        points += round(row['rush_yards'], 0)*0.1
        points += 3 if row['rush_yards'] > 100 else 0
        # rush_touchdowns
        points += round(row['rush_touchdowns'], 0)*6
        # receptions
        points += round(row['receptions'], 0)
        # receiving_yards
        points += round(row['receiving_yards'], 0)*0.1
        points += 3 if row['receiving_yards'] > 100 else 0
        # receiving_touchdowns
        points += round(row['receiving_touchdowns'], 0)*6
        return points
    def addPoints(self, source: pd.DataFrame, cd: pd.DataFrame):
        new_df = pd.DataFrame(columns=list(source.columns)+['points'])
        for index, row in source.iterrows():
            pid = row['p_id']
            wy = row['wy']
            position = row['position']
            stats = cd.loc[(cd['p_id']==pid)&(cd['wy']==wy)].squeeze()
            if position == 'QB':
                points = self.getQbPoints(stats)
            else:
                points = self.getSkillPoints(stats)
            new_df.loc[len(new_df.index)] = list(row.values) + [points]
        return new_df
    def buildTarget(self, source: pd.DataFrame):
        if 'target.csv' in os.listdir(self._dir):
            print('target.csv already created.')
            self.setTarget()
            return
        print('Creating target...')
        num_cores = multiprocessing.cpu_count()-1
        num_partitions = num_cores
        source_split = np.array_split(source, num_partitions)
        df_list = []
        if __name__ == 'fantasyPredictions.build':
            pool = multiprocessing.Pool(num_cores)
            all_dfs = pd.concat(pool.map(partial(self.addPoints, cd=self.cd), source_split))
            df_list.append(all_dfs)
            pool.close()
            pool.join()
            new_df = pd.concat(df_list)
            positions = ['QB', 'RB', 'WR', 'TE']
            wys = list(OrderedSet(source['wy'].values))
            df_list = []
            for wy in wys:
                for position in positions:
                    temp_df: pd.DataFrame = new_df.loc[(new_df['position']==position)&(new_df['wy']==wy)]
                    temp_df.sort_values(by=['points'], ascending=False, inplace=True)
                    temp_df.reset_index(drop=True, inplace=True)
                    temp_df['week_rank'] = temp_df.index
                    df_list.append(temp_df)
            new_df = new_df.merge(pd.concat(df_list), on=list(new_df.columns), how='left')
            self.saveFrame(new_df, "target")
            self.target = new_df
        return
    def buildPredTargets(self, source: pd.DataFrame):
        if 'pred_targets.csv' in os.listdir(self._dir):
            print('pred_targets already exists.')
            self.pred_targets = pd.read_csv("%s.csv" % (self._dir + "pred_targets"))
            return
        print('Creating pred_targets...')
        df_list = []
        for position in self.positions:
            t_cols = self.pred_target_stats[position]
            sf = source.loc[source['position']==position, ['key', 'p_id']]
            df: pd.DataFrame = self.pos_data[position]
            df = df[['game_key', 'p_id']+t_cols]
            df.columns = ['key', 'p_id']+t_cols
            sf = sf.merge(df, on=['key', 'p_id'])
            df_list.append(sf)
        new_df = pd.concat(df_list)
        source = source.merge(new_df, on=['key', 'p_id'])
        self.saveFrame(source, "pred_targets")
        self.pred_targets = source
        return
    def joinAll(self, source: pd.DataFrame):
        print('Joining...')
        new_df = source.copy()
        trainInfo = pd.DataFrame(columns=['num', 'name'])
        num = 1
        for fn in os.listdir(self.features_dir):
            try:
                if 'source' not in fn:
                    if re.search(r".*[N]$", fn): # fn ends with capital N
                        filename = [f for f in os.listdir(self.combineDir(fn)) if fn in f and '.csv' in f][0]
                        df = pd.read_csv(self.combineDir(fn) + filename)
                    else:
                        df = pd.read_csv("%s.csv" % (self.combineDir(fn) + fn))
                    new_df = new_df.merge(df, on=list(source.columns), how='left')
                    trainInfo.loc[len(trainInfo.index)] = [num, fn]
                    num += 1
            except FileNotFoundError:
                print(fn, 'not created.')
        self.saveFrame(trainInfo, "trainInfo")
        self.saveFrame(new_df, "train")
        self.train = new_df
        self.trainInfo = trainInfo
        return
    def joinTest(self, source: pd.DataFrame, df_list: list):
        print('Joining test...')
        # sort df_list _types to match features directory order
        _types = [t for _, t in df_list]
        fns: list = os.listdir(self.features_dir)
        [fns.remove(fn) for fn in fns if fn not in _types]
        _all = [(fns.index(f), f, df) for df, f in df_list]
        _all.sort(key=lambda x: x[0])
        new_df = source.copy()
        for _, _, df in _all:
            new_df = new_df.merge(df, on=list(source.columns), how='left')
        self.saveFrame(new_df, "test")
        self.test = new_df
        return
    def getOlsDrops(self, X, y, OLS_THRESHOLD):
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
            if val > OLS_THRESHOLD and name != 'const':
                drops.append(name)
        return drops
    def createPredTrain_positions(self):
        print('Creating pred_train and models...')
        self.setTrain()
        self.setPredTargets()
        df_list = []
        predInfo = pd.DataFrame(columns=['position', 'num', 'name'])
        for position in self.positions:
            t_cols = self.pred_target_stats[position]
            train = self.train.loc[self.train['position']==position]
            targets = self.pred_targets.loc[self.pred_targets['position']==position]
            count = 1
            for col in t_cols:
                X = train.drop(columns=self.str_cols)
                y = targets[col]
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                pickle.dump(scaler, open((self.models_dir + position + '_pred_' + col + '-scaler.sav'), 'wb'))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                models = self.pred_target_models[position][col]
                for name in models:
                    model: LinearRegression = models[name]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        model.fit(X_train, y_train)
                        acc = model.score(X_test, y_test)
                        print(f"Position: {position}, Target: {col}, {name} Accuracy: {acc}")
                        preds = model.predict(X)
                        col_name = position + '_pred_' + col + '_' + name
                        train[col_name] = preds
                        pickle.dump(model, open((self.models_dir + col_name + '.sav'), 'wb'))
                        predInfo.loc[len(predInfo.index)] = [position, count, col_name]
                        count += 1
            df_list.append(train)
        df = pd.concat(df_list)
        df = self.train[self.str_cols].merge(df, on=self.str_cols)
        self.saveFrame(df, "pred_train")
        self.saveFrame(predInfo, 'pred_info')
        self.pred_train = df
        # -----------------------
        self.setTrain()
        self.setTarget()
        df = pd.read_csv("%s.csv" % (self._dir + "pred_train"))
        for position in self.positions:
            drop_cols = [col for col in df.columns if col not in self.train.columns and position not in col]
            X = df.loc[df['position']==position].drop(columns=drop_cols+self.str_cols)
            target = self.target.loc[self.target['position']==position]
            for t_col in self.position_all_models:
                y = target[t_col]
                scaler = StandardScaler()
                X: pd.DataFrame = scaler.fit_transform(X)
                pickle.dump(scaler, open((self.models_dir + position + '_pred_' + t_col + '-scaler.sav'), 'wb'))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                for modelName in self.position_all_models[t_col]:
                    model: LinearRegression = self.position_all_models[t_col][modelName]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        if modelName in self.non_cont_models:
                            y_train = y_train.round(0)
                            y_test = y_test.round(0)
                        model.fit(X_train, y_train)
                        acc = model.score(X_test, y_test)
                        print(f"Position: {position}, Target: {t_col}, Model: {modelName}, Accuracy: {acc}")
                        pickle.dump(model, open((self.models_dir + position + '_pred_' + t_col + '_' + modelName + '.sav'), 'wb'))
        return
    def saveModels(self, showPreds: bool):
        # clear old models
        [os.remove(self._dir + 'models/' + fn) for fn in os.listdir(self._dir + 'models/')]
        train = self.train
        target = self.target
        t_cols = list(target.columns[len(self.str_cols):])
        drops_df = pd.DataFrame()
        for col in t_cols:
            threshold = self.target_ols_thresholds[col]
            data = train.merge(target[self.str_cols+[col]], on=self.str_cols, how='left')
            # if col == 'points': # remove outliers
            #     print("Data Shape before outliers:", data.shape)
            #     data: pd.DataFrame = data.loc[data[col].between(5, 40)]
            #     print("Data Shape after outliers:", data.shape)
            X = data.drop(columns=self.str_cols+[col])
            y = data[col]
            drops = self.getOlsDrops(X, y, threshold)
            drops_df[col] = [','.join(drops)]
            # print(col, len(drops))
            X = X.drop(drops, axis=1)
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            joblib.dump(scaler, open((self._dir + 'models/' + col + '-scaler.sav'), 'wb'))
            models = self.all_models[col]
            for name in models:
                model = models[name]
                if name == 'log' or name == 'forest':
                    y_train = y_train.round(0)
                    y_test = y_test.round(0)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                print(name + '_accuracy - ' + col + ':', acc)
                joblib.dump(model, open((self._dir + 'models/' + col + '_' + name + '.sav'), 'wb'))
                if showPreds:
                    preds = model.predict(X_test)
                    for i in range(10):
                        p = str(int(preds[i]))
                        e = str(y_test.values[i])
                        print('P: ' + p + ' <=> E: ' + e)
        print()
        self.saveFrame(drops_df, "drops")
        print('Drops created.')    
        print('Models saved.')
        return
    def saveModels_positions(self):
        # t_cols = list(self.target.columns[len(STR_COLS):])
        t_cols = ['points']
        drops_df = pd.DataFrame()
        for position in self.positions:
            pos_train: pd.DataFrame = self.train.loc[self.train['position']==position]
            pos_target: pd.DataFrame = self.target.loc[self.target['position']==position]
            for col in t_cols:
                threshold = self.target_ols_thresholds[col]
                data = pos_train.merge(pos_target[self.str_cols+[col]], on=self.str_cols, how='left')
                X = data.drop(columns=self.str_cols+[col])
                y = data[col]
                drops = self.getOlsDrops(X, y, threshold)
                drops_df[position + "_" + col] = [','.join(drops)]
                X = X.drop(drops, axis=1)
                scaler = StandardScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                joblib.dump(scaler, open((self._dir + 'models/' + position + '_' + col + '-scaler.sav'), 'wb'))
                models = self.position_all_models[col]
                for name in models:
                    model: LinearRegression = models[name]
                    if name in self.non_cont_models:
                        y_train = y_train.round(0)
                        y_test = y_test.round(0)
                    model.fit(X_train, y_train)
                    if name not in self.metric_models:
                        acc = model.score(X_test, y_test)
                    else:
                        acc = accuracy_score(y_test, model.predict(X_test))
                    print(position + '_' + name + '_accuracy - ' + col + ':', acc)
                    joblib.dump(model, open((self._dir + 'models/' + position + '_' + col + '_' + name + '.sav'), 'wb'))
        self.saveFrame(drops_df, "drops_positions")
        print("Position drops created.")   
        print('Individual position models saved.')    
        return
    def predict(self):
        self.setTarget()
        self.setDrops()
        print('Predicting...')
        t_cols = list(self.target.columns[len(self.str_cols):])
        X = self.test.drop(columns=self.str_cols)
        all_names = []
        for t_name in t_cols:
            if type(self.drops_df[t_name].values[0]) is str:
                drops = self.drops_df[t_name].values[0].split(',')
                X1 = X.drop(drops, axis=1)
            else:
                X1 = X.copy()
            scaler: StandardScaler = joblib.load(open((self.models_dir + t_name + '-scaler.sav'), 'rb'))
            X_scaled = scaler.transform(X1)
            fns = [
                fn for fn in os.listdir(self.models_dir) 
                if t_name in fn and 
                'scaler' not in fn and 
                fn.split("_")[0] not in ['QB', 'RB', 'WR', 'TE'] and 
                'pred' not in fn
            ]
            for fn in fns:
                modelName = fn.replace('.sav','')
                print('Model name:', modelName)
                model = joblib.load(open((self.models_dir + fn), 'rb'))
                preds = model.predict(X_scaled)
                self.test[modelName] = preds
                all_names.append(modelName)
        self.test = self.test.round(2)
        self.test = self.test[self.str_cols+all_names]
        sort_col = [n for n in all_names if 'points' in n][0]
        self.test.sort_values(by=[sort_col], ascending=False, inplace=True)
        self.saveFrame(self.test, "predictions")
        # save each position predictions
        for pos in self.positions:
            temp_df = self.test.loc[self.test['position']==pos]
            temp_df.sort_values(by=[sort_col], ascending=False, inplace=True)
            self.saveFrame(temp_df, ("predictions_" + pos))
        return
    def predict_positions(self):
        self.setTarget()
        self.setDropsPositions()
        print('Predicting individual positions...')
        # t_cols = list(self.target.columns[len(STR_COLS):])
        t_cols = ['points']
        all_preds = { pos: [] for pos in self.positions }
        for position in self.positions:
            X = self.test.loc[self.test['position']==position]
            X_copy = X[self.str_cols]
            X = X.drop(columns=self.str_cols)
            for t_name in t_cols:
                print(f"Model params: {position} - {t_name}")
                drops_col = position + "_" + t_name
                # if type(drops_df[drops_col].values[0]) is str:
                drops = self.drops_positions_df[drops_col].values[0].split(',')
                X1 = X.drop(drops, axis=1)
                # else:
                #     X1 = X.copy()
                scaler: StandardScaler = joblib.load(open((self.models_dir + position + '_' + t_name + '-scaler.sav'), 'rb'))
                X_scaled = scaler.transform(X1)
                fns = [fn for fn in os.listdir(self.models_dir) if t_name in fn and 'scaler' not in fn and position in fn and 'pred' not in fn]
                for fn in fns:
                    modelName = fn.replace('.sav','')
                    print('Model name:', modelName)
                    model = joblib.load(open((self.models_dir + fn), 'rb'))
                    preds = model.predict(X_scaled)
                    X_copy[modelName] = preds
            all_preds[position].append(X_copy)
        for position in self.positions:
            frames = all_preds[position]
            df = reduce(lambda x, y: pd.merge(x, y, on=self.str_cols), frames)
            sort_col = [col for col in df.columns if 'points' in col and position in col][0]
            df = df.round(2)
            fdf = pd.read_csv("%s.csv" % (self._dir + 'predictions_' + position))
            fdf = fdf.merge(df, on=self.str_cols)
            fdf.sort_values(by=[sort_col], ascending=False, inplace=True)
            self.saveFrame(fdf, ("predictions_" + position))
        return
    def predTrainPredict_positions(self):
        self.setTest()
        predInfo = pd.read_csv("%s.csv" % (self._dir + "pred_info"))
        pred_cols, df_list = [], []
        for position in self.positions:
            df = self.test.loc[self.test['position']==position]
            p_info = predInfo.loc[predInfo['position']==position]
            for modelName in p_info['name'].values:
                col = '_'.join(modelName.split("_")[:-1])
                X = df.drop(columns=self.str_cols)
                scaler: StandardScaler = pickle.load(open((self.models_dir + col + '-scaler.sav'), 'rb'))
                X = scaler.transform(X)
                print(f'Pred model name: {modelName}')
                model: LinearRegression = pickle.load(open((self.models_dir + modelName + '.sav'), 'rb'))
                preds = model.predict(X)
                df[modelName] = preds
                pred_cols.append(modelName)
            df_list.append(df)
        new_df = pd.concat(df_list)
        new_df = self.test[self.str_cols].merge(new_df, on=self.str_cols)
        self.saveFrame(new_df, 'pred_test')
        # all_models predictions
        t_cols = ['points']
        pred_cols_1, df_list = [], []
        for position in self.positions:
            drop_cols = [col for col in new_df.columns if col not in self.test.columns and position not in col]
            X = new_df.loc[new_df['position']==position]
            X_copy = X[self.str_cols]
            X = X.drop(columns=self.str_cols+drop_cols)
            for col in t_cols:
                scaler: StandardScaler = pickle.load(open((self.models_dir + position + '_pred_' + col + '-scaler.sav'), 'rb'))
                X = scaler.transform(X)
                fns = [fn for fn in os.listdir(self.models_dir) if position in fn and 'pred' in fn and col in fn and 'scaler' not in fn]
                for fn in fns:
                    modelName = fn.replace('.sav','')
                    print('pred model name:', modelName)
                    model: LogisticRegression = pickle.load(open((self.models_dir + fn), 'rb'))
                    preds = model.predict(X)
                    X_copy['pred_' + col + '_' + modelName] = preds
                pred_cols_1.append('pred_' + col + '_' + modelName)
            df_list.append(X_copy)
        new_df = pd.concat(df_list)
        new_df = new_df.sort_values(by=[pred_cols_1[0]], ascending=False)
        new_df = new_df.round(0)
        self.saveFrame(new_df, "pred_predictions")
        for position in self.positions:
            temp_df = new_df.loc[new_df['position']==position].dropna(axis=1)
            fdf = pd.read_csv("%s.csv" % (self._dir + "predictions_" + position))
            pt = pd.read_csv("%s.csv" % (self._dir + "pred_test"))
            pt_cols = [col for col in pred_cols if position in col]
            pt = pt[self.str_cols+pt_cols]
            fdf = fdf.merge(temp_df, on=self.str_cols)
            fdf = fdf.merge(pt, on=self.str_cols)
            fdf = fdf.round(1)
            fdf.sort_values(by=[[col for col in pred_cols_1 if position in col][0]], ascending=False, inplace=True)
            self.saveFrame(fdf, ("predictions_" + position))
        return
    def setTrain(self):
        self.train = pd.read_csv("%s.csv" % (self._dir + "train"))
        return
    def setTarget(self):
        self.target = pd.read_csv("%s.csv" % (self._dir + "target"))
        return
    def setTest(self):
        self.test = pd.read_csv("%s.csv" % (self._dir + "test"))
        return
    def setDrops(self):
        self.drops_df = pd.read_csv("%s.csv" % (self._dir + "drops"))
        return
    def setDropsPositions(self):
        self.drops_positions_df = pd.read_csv("%s.csv" % (self._dir + "drops_positions"))
        return
    def setPredTargets(self):
        self.pred_targets = pd.read_csv("%s.csv" % (self._dir + "pred_targets"))
        return
    def main(self):
        """
        Calls each feature function, joins to train.csv, \n
        and writes models.
        """
        # build source if it does not exist
        f_type = 'source'
        source = buildSource(self.cd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build target if it does not exist
        self.buildTarget(source)
        print()
        # ---------------------------------------------
        # build target if it does not exist
        self.buildPredTargets(source.copy())
        print()
        # ---------------------------------------------
        # build seasonAvg
        f_type = 'seasonAvg'
        buildSeasonAvg(source.copy(), self.fd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build encodedPosition
        f_type = 'encodedPosition'
        buildEncodedPosition(source.copy(), self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build maxWeekRank
        f_type = 'maxWeekRank'
        buildMaxWeekRank(source.copy(), self.fd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build pointsN
        f_type = 'pointsN'
        buildPointsN(5, source.copy(), self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build careerAvg
        f_type = 'careerAvg'
        buildCareerAvg(source.copy(), self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build allowedPointsN
        f_type = 'allowedPointsN'
        buildAllowedPointsN(5, source.copy(), self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build season rankings
        f_type = 'seasonRankings'
        buildSeasonRankings(source.copy(), self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build isPrimary
        f_type = 'isPrimary'
        ip = IsPrimary(self.combineDir(f_type))
        ip.buildIsPrimary(source.copy(), self.ocd)
        print()
        # ---------------------------------------------
        # build maddenRatings
        f_type = 'maddenRatings'
        mr = MaddenRatings(self.combineDir(f_type))
        mr.buildMaddenRatings(source.copy(), self.rdf, False)
        print()
        # ---------------------------------------------
        # build isStarter
        f_type = 'isStarter'
        iss = IsStarter(self.combineDir(f_type))
        iss.buildIsStarter(source.copy(), self.sdf, False)
        print()
        # ---------------------------------------------
        # lastStatsN
        f_type = "lastStatsN"
        lsn = LastStatsN(self.pos_data, self.combineDir(f_type))
        lsn.buildLastStatsN(10, source.copy(), False)
        print()
        # ---------------------------------------------
        # build avgStatsN if does not exist
        f_type = 'avgStatsN'
        asn = AvgStatsN(self.pos_data, self.combineDir(f_type))
        asn.buildAvgStatsN_parallel(5, source.copy())
        print()
        # ---------------------------------------------
        # build advancedStats if does not exist
        f_type = 'advancedStats'
        ads = AdvancedStats(self.adf, self.combineDir(f_type))
        ads.buildAdvancedStats(source.copy(), False)
        print()
        # ---------------------------------------------
        # build seasonAvgSnapPercentages if does not exist
        f_type = 'seasonAvgSnapPercentages'
        sasp = SeasonAvgSnapPercentages(self.scdf, self.combineDir(f_type))
        sasp.build(source.copy(), False)
        print()
        # ---------------------------------------------
        # build lastSnapPercentagesN if does not exist
        f_type = 'lastSnapPercentagesN'
        lsp = LastSnapPercentagesN(self.scdf, self.combineDir(f_type))
        lsp.build(5, source.copy(), False)
        print()
        # ---------------------------------------------
        # build lastSnapDifferencesN if does not exist
        f_type = 'lastSnapDifferencesN'
        lsd = LastSnapDifferencesN(self.scdf, self.combineDir(f_type))
        lsd.build(11, source.copy(), False)
        print()
        # ---------------------------------------------
        self.joinAll(source)
        print()
        # ---------------------------------------------
        self.setTrain()
        self.setTarget()
        self.saveModels(False)
        print()
        # ---------------------------------------------
        self.saveModels_positions()
        print()
        # # ---------------------------------------------
        # # use train to create all pred_targets predictions and join
        # self.createPredTrain_positions()
        # print()
        return
    def new_main(self, week: int, year: int):
        wy = str(week) + ' | ' + str(year)
        # curr starters
        if wy in self.sdf['wy'].values:
            sdf = self.sdf.loc[self.sdf['wy']==wy]
        else:
            s_dir = 'starters_' + str(year)[-2:] + '/'
            sdf = pd.read_csv("%s.csv" % (self.data_path + s_dir + "starters_w" + str(week)))
        # build new source/matchups for given week and year
        f_type = 'source'
        source = buildNewSource(week, year, self.fd, sdf, self.combineDir(f_type))
        print()
        # declare list to store each feature dataframe
        df_list = []
        # ---------------------------------------------
        # build seasonAvg
        f_type = 'seasonAvg'
        df_list.append((buildNewSeasonAvg(source.copy(), self.fd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build encodedPosition
        f_type = 'encodedPosition'
        df_list.append((buildNewEncodedPosition(source.copy(), self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build maxWeekRank
        f_type = 'maxWeekRank'
        df_list.append((buildNewMaxWeekRank(source.copy(), self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build pointsN
        f_type = 'pointsN'
        df_list.append((buildNewPointsN(5, source.copy(), self.fd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build careerAvg
        f_type = 'careerAvg'
        df_list.append((buildNewCareerAvg(source.copy(), self.fd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build allowedPointsN
        f_type = 'allowedPointsN'
        df_list.append((buildNewAllowedPointsN(5, source.copy(), self.fd, self.gd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build seasonRankings
        f_type = 'seasonRankings'
        df_list.append((buildNewSeasonRankings(source.copy(), self.gd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build isPrimary
        f_type = 'isPrimary'
        ip = IsPrimary(self.combineDir(f_type))
        df_list.append((ip.buildNewIsPrimary(source.copy(), self.ocd), f_type))
        print()
        # ---------------------------------------------
        # build maddenRatings
        f_type = 'maddenRatings'
        mr = MaddenRatings(self.combineDir(f_type))
        df_list.append((mr.buildMaddenRatings(source.copy(), self.rdf, True), f_type))
        print()
        # ---------------------------------------------
        # build isStarter
        f_type = 'isStarter'
        iss = IsStarter(self.combineDir(f_type))
        df_list.append((iss.buildIsStarter(source.copy(), sdf, True), f_type))
        print()
        # ---------------------------------------------
        # build lastStatsN
        f_type = "lastStatsN"
        lsn = LastStatsN(self.pos_data, self.combineDir(f_type))
        df_list.append((lsn.buildLastStatsN(10, source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build avgStatsN
        f_type = 'avgStatsN'
        asn = AvgStatsN(self.pos_data, self.combineDir(f_type))
        df_list.append((asn.buildAvgStatsN(5, source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build advancedStats
        f_type = 'advancedStats'
        ads = AdvancedStats(self.adf, self.combineDir(f_type))
        df_list.append((ads.buildAdvancedStats(source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build seasonAvgSnapPercentages
        f_type = 'seasonAvgSnapPercentages'
        sasp = SeasonAvgSnapPercentages(self.scdf, self.combineDir(f_type))
        df_list.append((sasp.build(source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build lastSnapPercentagesN
        f_type = 'lastSnapPercentagesN'
        lsp = LastSnapPercentagesN(self.scdf, self.combineDir(f_type))
        df_list.append((lsp.build(5, source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build lastSnapDifferencesN
        f_type = 'lastSnapDifferencesN'
        lsd = LastSnapDifferencesN(self.scdf, self.combineDir(f_type))
        df_list.append((lsd.build(11, source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # merge test
        self.joinTest(source, df_list)
        print()
        # ---------------------------------------------
        # check test has same features as train.csv
        self.setTrain()
        if self.train.shape[1] != self.test.shape[1]:
            print('Train shape: ' + str(self.train.shape[1]) + " != Test shape:" + str(self.test.shape[1]))
            return
        # ---------------------------------------------
        # make predictions
        self.setTest()
        self.predict()
        print()
        # ---------------------------------------------
        # make position predictions
        self.setTest()
        self.predict_positions()
        print()
        # # ---------------------------------------------
        # # make pred_train predictions
        # self.predTrainPredict_positions()
        # print()
        return
    
# END / Build