import pandas as pd
import numpy as np
import os
import pickle
import regex as re
import statsmodels.api as sm
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import NuSVC, NuSVR, LinearSVC, LinearSVR

try:
    from gamePredictions.features.source.source import buildSource, buildSourceIndividual, buildNewSource
    from gamePredictions.features.teamElos.teamElos import Elos
    from gamePredictions.features.seasonInfo.seasonInfo import buildSeasonInfo, buildNewSeasonInfo
    from gamePredictions.features.pred_standings.tiebreakerAttributes import TiebreakerAttributes
    from gamePredictions.features.pred_standings.predStandings import PredStandings
    from gamePredictions.features.avgN.avgN import buildAvgN, buildNewAvgN
    from gamePredictions.features.qbHalfGame.qbHalfGame import buildQbHalfGame, buildNewQbHalfGame
    from gamePredictions.features.lastWonN.lastWonN import buildLastWonN, buildNewLastWonN
    from gamePredictions.features.pointsAllowedN.pointsAllowedN import buildPointsAllowedN, buildNewPointsAllowedN
    from gamePredictions.features.playerGrades.playerGrades import PlayerGrades
    from gamePredictions.features.playerGrades.models import PgModels
    from gamePredictions.features.seasonRankings.seasonRankings import buildSeasonRankings, buildNewSeasonRankings
    from gamePredictions.features.pointsN.pointsN import buildPointsN, buildNewPointsN
    from gamePredictions.features.vegaslines.vegaslines import buildVegasLine, buildNewVegasLine
    from gamePredictions.features.matchupInfo.matchupInfo import buildMatchupInfo, buildNewMatchupInfo
    from gamePredictions.features.coachElos.coachElos import CoachElos
    from gamePredictions.features.starterInfo.starterInfo import StarterInfo
    from gamePredictions.features.starterMaddenRatings.starterMaddenRatings import StarterMaddenRatings
    from gamePredictions.features.lastGameStatsN.lastGameStatsN import LastGameStatsN
    from gamePredictions.features.lastOppWonN.lastOppWonN import LastOppWonN
    from gamePredictions.features.summariesAvgN.summariesAvgN import SummariesAvgN
    from gamePredictions.features.advancedStarterStats.advancedStarterStats import AdvancedStarterStats
    from gamePredictions.features.short_positionGroupSeasonAvgSnapPcts.positionGroupSeasonAvgSnapPcts import PositionGroupSeasonAvgSnapPcts
    from gamePredictions.features.overUnders.overUnders import OverUnders
except ModuleNotFoundError as error:
    print(error)
    
class Build:
    def __init__(self, all_paths: dict, _dir: str):
        self.all_paths = all_paths
        self._dir = _dir
        self.features_dir = _dir + 'features/'
        self.models_dir = _dir + 'models/'
        self.positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        # paths
        self.data_path = self.all_paths['dp']
        self.position_path = self.all_paths['pp']
        self.starters_path = self.all_paths['sp']
        self.team_names_path = self.all_paths['tnp']
        self.coaches_path = self.all_paths['cp']
        self.madden_path = self.all_paths['mrp']
        self.snap_path = self.all_paths['sc']
        # dataframes
        self.cd = pd.read_csv("%s.csv" % (self.data_path + "gameData"))
        self.ocd = pd.read_csv("%s.csv" % (self.data_path + "oldGameData_78"))
        self.qdf = pd.read_csv("%s.csv" % (self.position_path + "QBData"))
        self.sl = pd.read_csv("%s.csv" % (self.data_path + "seasonLength"))
        self.sdf = pd.read_csv("%s.csv" % (self.starters_path + "allStarters"))
        self.tn = pd.read_csv("%s.csv" % (self.team_names_path + "teamNames_line"))
        self.cdf = pd.read_csv("%s.csv" % (self.coaches_path + "coachInfo"))
        self.rdf = pd.read_csv("%s.csv" % (self.madden_path + "playerRatings"))
        self.sum_df = pd.read_csv("%s.csv" % (self.data_path + "summaries"))
        self.adf = pd.read_csv("%s.csv" % (self.data_path + "advancedStats"))
        self.scdf = pd.read_csv("%s.csv" % (self.snap_path + "snap_counts"))
        self.position_frames: dict = {
            pos: (pd.read_csv("%s.csv" % (self.position_path + pos + "Data")) if pos not in ['LB', 'DL'] else pd.read_csv("%s.csv" % (self.position_path + "LBDLData")))
            for pos in self.positions
        }
        # more frames
        self.train: pd.DataFrame = None
        self.target: pd.DataFrame = None
        self.pred_train: pd.DataFrame = None
        self.pred_targets: pd.DataFrame = None
        self.trainInfo: pd.DataFrame = None
        self.drops_df: pd.DataFrame = None
        self.test: pd.DataFrame = None
        # models stuff
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        self.target_ols_thresholds = {
            'away_points': 0.25, 'home_points': 0.3, 'home_won': 0.2,
        }
        self.all_models = {
            'won': {
                'log': LogisticRegression(n_jobs=-1, max_iter=5000),
                'forest': RandomForestClassifier(n_jobs=-1)
            },
            'points': {
                'linear': LinearRegression(n_jobs=-1)
            }
        }
        # pred targets info
        self.pred_stat_cols = ['home_points', 'away_points']
        self.pred_target_models = {
            'home_points': {
                'forest': RandomForestClassifier(n_jobs=-1)
            },
            'away_points': {
                'forest': RandomForestClassifier(n_jobs=-1)
            },
            'home_tds': {
                'forest': RandomForestClassifier(n_jobs=-1)
            },
            'away_tds': {
                'forest': RandomForestClassifier(n_jobs=-1)
            },
            'home_fgs': {
                'forest': RandomForestClassifier(n_jobs=-1)
            },
            'away_fgs': {
                'forest': RandomForestClassifier(n_jobs=-1)
            },
        }
        self.short_features = [
            'positionGroupSeasonAvgSnapPcts'
        ]
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
    def combineDir(self, _type, isShort=False):
        """
        Combines feature dir with feature type dir \n
        Args:
            _type (_type_): feature name/type
        Returns:
            _type_: feature root _dir + specific feature name
        """
        prefix = 'short_' if isShort else ''
        return self.features_dir + prefix +  _type + '/'
    def buildTarget(self, source: pd.DataFrame):
        if 'target.csv' in os.listdir(self._dir):
            print('Target already exists.')
            self.target = pd.read_csv("%s.csv" % (self._dir + "target"))
            return
        new_df = pd.DataFrame(columns=list(source.columns)+['home_won', 'home_points', 'away_points'])
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), 'Target')
            key = row['key']
            wy = row['wy']
            home_abbr = row['home_abbr']
            away_abbr = row['away_abbr']
            winning_abbr, home_points, away_points = self.cd.loc[
                self.cd['key']==key, ['winning_abbr', 'home_points', 'away_points']
            ].values[0]
            home_won = 0
            if winning_abbr == home_abbr:
                home_won = 1
            new_df.loc[len(new_df.index)] = list(row.values) + [home_won, home_points, away_points]
        self.saveFrame(new_df, 'target')
        self.target = new_df
        return
    def buildPredTargets(self, source: pd.DataFrame):
        if 'pred_targets.csv' in os.listdir(self._dir):
            print('pred_targets already exists.')
            self.pred_targets = pd.read_csv("%s.csv" % (self._dir + "pred_targets"))
            return
        print('Creating pred_targets...')
        cd = self.cd[self.str_cols+self.pred_stat_cols]
        source = source.merge(cd, on=self.str_cols)
        source['home_tds'] = source['home_points'].apply(lambda x: int(x/7))
        source['home_fgs'] = source.apply(lambda x: int((x['home_points'] - (x['home_tds']*7)) / 3), axis=1)
        source['away_tds'] = source['away_points'].apply(lambda x: int(x/7))
        source['away_fgs'] = source.apply(lambda x: int((x['away_points'] - (x['away_tds']*7)) / 3), axis=1)
        self.saveFrame(source, 'pred_targets')
        self.pred_targets = source
        return
    def joinAll(self, source: pd.DataFrame, isShort: bool):
        print('Joining short...') if isShort else print('Joining...')
        new_df = source.copy()
        if isShort:
            start = new_df.loc[new_df['wy'].str.contains('2012')].index.values[0]
            new_df = new_df.loc[new_df.index>=start]
        trainInfo = pd.DataFrame(columns=['num', 'name'])
        num = 1
        for fn in os.listdir(self.features_dir):
            try:
                if 'source' not in fn and (isShort or (not isShort and 'short_' not in fn)):
                    if 'playerGrades' in fn:
                        pg_fns = os.listdir(self.combineDir(fn))
                        for pg_fn in pg_fns:
                            if re.search(r"[a-z][a-z]_playerGrades.csv", pg_fn):
                                df = pd.read_csv(self.combineDir(fn) + pg_fn)
                                new_df = new_df.merge(df, on=list(source.columns), how='left')
                                trainInfo.loc[len(trainInfo.index)] = [num, pg_fn.replace('.csv','')]
                                num += 1
                    elif 'advancedStarterStats' in fn:
                        pg_fns = os.listdir(self.combineDir(fn))
                        for pg_fn in pg_fns:
                            if re.search(r"[a-z][a-z]_advancedStarterStats.csv", pg_fn):
                                df = pd.read_csv(self.combineDir(fn) + pg_fn)
                                new_df = new_df.merge(df, on=list(source.columns), how='left')
                                trainInfo.loc[len(trainInfo.index)] = [num, pg_fn.replace('.csv','')]
                                num += 1
                    else:
                        if re.search(r".*[N]$", fn): # fn ends with capital N
                            filename = [f for f in os.listdir(self.combineDir(fn)) if fn in f and '.csv' in f][0]
                            filename = filename.replace('short_','')
                            df = pd.read_csv(self.combineDir(fn) + filename)
                        else:
                            df = pd.read_csv("%s.csv" % (self.combineDir(fn) + fn.replace('short_','')))
                        new_df = new_df.merge(df, on=list(source.columns), how='left')
                        trainInfo.loc[len(trainInfo.index)] = [num, fn]
                        num += 1
            except FileNotFoundError:
                print()
                print(fn, 'not created.')
        suffix = '_short' if isShort else ''
        trainInfo.to_csv("%s.csv" % (self._dir + "trainInfo" + suffix), index=False)
        new_df.to_csv("%s.csv" % (self._dir + "train" + suffix), index=False)
        self.train = new_df
        self.trainInfo = trainInfo
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
    def createPredTrain(self):
        print('Creating pred_train and models...')
        self.setTrain()
        self.setPredTargets()
        df = self.train.copy()
        t_cols = self.pred_targets.columns[len(self.str_cols):]
        predInfo = pd.DataFrame(columns=['num', 'name'])
        count = 1
        for col in t_cols:
            X = self.train.drop(columns=self.str_cols)
            y = self.pred_targets[col]
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            pickle.dump(scaler, open((self.models_dir + 'pred_' + col + '-scaler.sav'), 'wb'))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models = self.pred_target_models[col]
            for name in models:
                model = models[name]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(X_train, y_train)
                    acc = model.score(X_test, y_test)
                    print(f"Target: {col}, Accuracy: {acc}")
                    preds = model.predict(X)
                    col_name = 'pred_' + col + '_' + name
                    df[col_name] = preds
                    pickle.dump(model, open((self.models_dir + col_name + '.sav'), 'wb'))
                    predInfo.loc[len(predInfo.index)] = [count, col_name]
                    count += 1
        self.saveFrame(df, "pred_train")
        self.saveFrame(predInfo, 'pred_info')
        self.pred_train = df
        # model for home_won using pred_train
        print('Creating pred_home_won model...')
        self.setTarget()
        X = df.drop(columns=self.str_cols)
        y = self.target['home_won']
        # threshold = self.target_ols_thresholds[col]
        # drops_df = pd.DataFrame()
        # drops = self.getOlsDrops(X, y, 0.02)
        # drops_df[col] = [','.join(drops)]
        # X = X.drop(columns=drops)
        scaler = StandardScaler()
        X: pd.DataFrame = scaler.fit_transform(X)
        pickle.dump(scaler, open((self.models_dir + 'pred_home_won-scaler.sav'), 'wb'))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(n_jobs=-1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"home_won - Accuracy: {acc}")
            pickle.dump(model, open((self.models_dir + 'pred_home_won.sav'), 'wb'))
        # self.saveFrame(drops_df, "pred_drops")
        # print('Drops saved.')
        return
    def saveModels(self, isShort: bool):
        self.setTrain()
        self.setTarget()
        self.train = pd.read_csv("%s.csv" % (self._dir + "train_short")) if isShort else self.train
        suffix = '_short' if isShort else ''
        # clear old models
        # [os.remove(self._dir + 'models/' + fn) for fn in os.listdir(self._dir + 'models/')]
        train = self.train
        target = self.target
        t_cols = list(target.columns[len(self.str_cols):])
        drops_df = pd.DataFrame()
        for col in t_cols:
            threshold = self.target_ols_thresholds[col]
            data = train.merge(target[self.str_cols+[col]], on=self.str_cols, how='left')
            X = data.drop(columns=self.str_cols+[col])
            y = data[col]
            drops = self.getOlsDrops(X, y, threshold)
            drops_df[col] = [','.join(drops)]
            X = X.drop(drops, axis=1)
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pickle.dump(scaler, open((self.models_dir + col + '-scaler' + suffix + '.sav'), 'wb'))
            target_type = col.split("_")[1]
            models = self.all_models[target_type]
            for name in models:
                model: LogisticRegression = models[name]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(X_train, y_train)
                    acc = model.score(X_test, y_test)
                    print(name + '_accuracy' + suffix + ' - ' + col + ':', acc)
                    pickle.dump(model, open((self.models_dir + col + '_' + name + suffix + '.sav'), 'wb'))   
        print()
        self.saveFrame(drops_df, "drops" + suffix)
        print('Drops created.')          
        print('Models saved.')
        return
    def joinTest(self, source: pd.DataFrame, df_list: list, isShort: bool):
        suffix = '_short' if isShort else ''
        print('Joining test' + suffix + '...')
        # sort df_list _types to match features directory order
        # _types = [t for _, t in df_list]
        fns: list = os.listdir(self.features_dir)
        # fns = [fn for fn in fns if 'short_' not in fn] if not isShort else fns
        # [fns.remove(fn) for fn in fns if fn not in _types]
        # print(fns)
        df_list = df_list if isShort else [(df, f) for df, f in df_list if f not in self.short_features]
        _all = []
        for df, f in df_list:
            try:
                _all.append((fns.index(f), f, df))
            except ValueError:
                if ('short_' + f) in fns:
                    _all.append((fns.index('short_' + f), f, df))
                continue
        _all.sort(key=lambda x: x[0])
        new_df = source.copy()
        for _, f, df in _all:
            new_df = new_df.merge(df, on=list(source.columns), how='left')
        new_df.to_csv("%s.csv" % (self._dir + "test" + suffix), index=False)
        self.test = new_df
        return
    def predict(self, isShort: bool):
        self.test = pd.read_csv("%s.csv" % (self._dir + "test_short")) if isShort else self.test
        suffix = '_short' if isShort else ''
        self.setTarget()
        self.setDrops()
        self.drops_df = pd.read_csv("%s.csv" % (self._dir + "drops_short")) if isShort else self.drops_df
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
            scaler: StandardScaler = pickle.load(open((self.models_dir + t_name + '-scaler' + suffix + '.sav'), 'rb'))
            X_scaled = scaler.transform(X1)
            fns = [fn for fn in os.listdir(self.models_dir) if t_name in fn and 'scaler' not in fn and 'pred' not in fn]
            fns = [fn for fn in fns if '_short' in fn] if isShort else [fn for fn in fns if '_short' not in fn]
            for fn in fns:
                modelName = fn.replace('.sav','')
                print('Model name:', modelName)
                model: LogisticRegression = pickle.load(open((self.models_dir + fn), 'rb'))
                preds = model.predict(X_scaled)
                self.test[modelName] = preds
                all_names.append(modelName)
                if modelName == 'home_won_log' or modelName == 'home_won_log_short':
                    print('Adding log probabilities..')
                    p_prob = model.predict_proba(X_scaled)
                    n0, n1 = 'away_win_probability' + suffix, 'home_win_probability' + suffix
                    self.test[n0] = p_prob[:, 0]
                    self.test[n1] = p_prob[:, 1]
                    all_names.append(n1)
                    all_names.append(n0)
        self.test = self.test.round(2)
        self.test = self.test[self.str_cols+all_names]
        point_cols = [col for col in self.test.columns if 'points' in col]
        point_cols = list(set(['_'.join(col.split("_")[1:]) for col in point_cols]))
        for col in point_cols:
            self.test[col + '_h_won'] = self.test.apply(lambda x: 1 if x['home_' + col] >= x['away_' + col] else 0, axis=1)
        if isShort:
            df = pd.read_csv("%s.csv" % (self._dir + "predictions"))
            # self.test.columns = self.str_cols + [(col + '_short') for col in self.test.columns if col not in self.str_cols]
            self.test = df.merge(self.test, on=self.str_cols)
        self.test.to_csv("%s.csv" % (self._dir + "predictions"), index=False)
        return
    def predTrainPredict(self):
        self.setTest()
        predInfo = pd.read_csv("%s.csv" % (self._dir + "pred_info"))
        df = self.test.copy()
        pred_cols = []
        for modelName in predInfo['name'].values:
            col = '_'.join(modelName.split("_")[:-1])
            X = self.test.drop(columns=self.str_cols)
            scaler: StandardScaler = pickle.load(open((self.models_dir + col + '-scaler.sav'), 'rb'))
            X = scaler.transform(X)
            print(f'Pred model name: {modelName}')
            model: RandomForestClassifier = pickle.load(open((self.models_dir + modelName + '.sav'), 'rb'))
            preds = model.predict(X)
            df[modelName] = preds
            pred_cols.append(modelName)
        self.saveFrame(df, 'pred_test')
        # load pred_home_won model
        print('Creating pred_predictions...')
        scaler: StandardScaler = pickle.load(open((self.models_dir + 'pred_home_won-scaler.sav'), 'rb'))
        X = df.drop(columns=self.str_cols)
        X = scaler.transform(X)
        fns = [fn for fn in os.listdir(self.models_dir) if 'pred_home_won' in fn and 'scaler' not in fn]
        for fn in fns:
            modelName = fn.replace('.sav','')
            print('pred model name:', modelName)
            model: LogisticRegression = pickle.load(open((self.models_dir + fn), 'rb'))
            preds = model.predict(X)
            df[modelName] = preds
            pred_cols.append(modelName)
        df = df[self.str_cols+pred_cols]
        self.saveFrame(df, 'pred_predictions')
        print('pred_predicitions created.')
        return
    def setTrain(self):
        self.train = pd.read_csv("%s.csv" % (self._dir + "train"))
        return
    def setTarget(self):
        self.target = pd.read_csv("%s.csv" % (self._dir + "target"))
        return
    def setPredTargets(self):
        self.pred_targets = pd.read_csv("%s.csv" % (self._dir + "pred_targets"))
        return
    def setPredTrain(self):
        self.pred_train = pd.read_csv("%s.csv" % (self._dir + "pred_train"))
        return
    def setTest(self):
        self.test = pd.read_csv("%s.csv" % (self._dir + "test"))
        return
    def setDrops(self):
        self.drops_df = pd.read_csv("%s.csv" % (self._dir + "drops"))
        return
    def main(self):
        """
        Calls each feature function, joins to train.csv, \n
        and writes models.
        """
        # build source if it does not exist
        f_type = 'source'
        source = buildSource(self.cd, self.combineDir(f_type))
        source_indiv = buildSourceIndividual(self.cd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build target
        self.buildTarget(source)
        print()
        # ---------------------------------------------
        # build pred targets
        self.buildPredTargets(source.copy())
        print()
        # ---------------------------------------------
        # build teamElos if it does not exist
        f_type = 'teamElos'
        elos = Elos(self.cd, self.combineDir(f_type))
        elos.build()
        elos.createBoth(source.copy(), isNew=False)
        print()
        # ---------------------------------------------
        # build seasonInfo if it does not exist
        f_type = 'seasonInfo'
        buildSeasonInfo(source.copy(), self.cd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build pred_standings if it does not exist
        f_type = 'pred_standings'
        tb = TiebreakerAttributes(self.cd, self.sl, self.combineDir(f_type))
        tb.buildAllTiebreakerAttributes()
        ps = PredStandings(self.sl, self.combineDir(f_type))
        ps.buildPredictions()
        ps.buildPredStandings(source.copy())
        print()
        # ---------------------------------------------
        # build avgN if it does not exist
        f_type = 'avgN'
        buildAvgN(5, source.copy(), self.ocd, self.target, drops_threshold=0.2, _dir=self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build qbHalfGame if it does not exist
        f_type = 'qbHalfGame'
        buildQbHalfGame(source.copy(), self.qdf, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build lastWonN if it does not exist
        f_type = 'lastWonN'
        buildLastWonN(20, source.copy(), self.ocd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build pointsAllowedN if it does not exist
        f_type = 'pointsAllowedN'
        buildPointsAllowedN(50, source.copy(), self.ocd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build playerGrades if it does not exist
        f_type = 'playerGrades'
        pg = PlayerGrades('qb', self.combineDir(f_type))
        pg.buildElos()
        pg.createBoth(source.copy(), self.sdf, isNew=False)
        print()
        # ---------------------------------------------
        # build seasonRankings if it does not exist
        f_type = 'seasonRankings'
        buildSeasonRankings(source.copy(), self.ocd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build pointsN if it does not exist
        f_type = 'pointsN'
        buildPointsN(50, source.copy(), self.ocd, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build vegaslines if it does not exist
        f_type = 'vegaslines'
        buildVegasLine(source.copy(), self.cd, self.tn, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build matchupInfo if it does not exist
        f_type = 'matchupInfo'
        buildMatchupInfo(source.copy(), self.cd, self.sl, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # build coachElos if it does not exist
        f_type = 'coachElos'
        c_elos = CoachElos(self.cd, self.cdf, self.combineDir(f_type))
        c_elos.build()
        c_elos.createBoth(source.copy(), isNew=False)
        print()
        # ---------------------------------------------
        # build playerGrades if it does not exist
        f_type = 'playerGrades'
        pg = PlayerGrades('rb', self.combineDir(f_type))
        pg.buildElos()
        pg.createBoth(source.copy(), self.sdf, isNew=False)
        print()
        # ---------------------------------------------
        # build playerGrades if it does not exist
        f_type = 'playerGrades'
        pg = PlayerGrades('wr', self.combineDir(f_type))
        pg.buildElos()
        pg.createBoth(source.copy(), self.sdf, isNew=False)
        print()
        # ---------------------------------------------
        # build starterInfo if it does not exist
        f_type = 'starterInfo'
        si = StarterInfo(self.combineDir(f_type))
        si.buildStarterInfo(source.copy(), False)
        print()
        # ---------------------------------------------
        # build starterMaddenRatings if it does not exist
        f_type = 'starterMaddenRatings'
        smr = StarterMaddenRatings(self.combineDir(f_type))
        smr.buildStarterMaddenRatings(source.copy(), self.sdf, self.rdf, False)
        print()
        # ---------------------------------------------
        # build lastGameStatsN if it does not exist
        f_type = 'lastGameStatsN'
        lgs = LastGameStatsN(self.cd.copy(), self.combineDir(f_type))
        lgs.buildLastGameStatsN(5, source.copy(), False)
        print()
        # ---------------------------------------------
        # build lastOppWonN if it does not exist
        f_type = 'lastOppWonN'
        low = LastOppWonN(self.cd.copy(), self.combineDir(f_type))
        low.buildLastOppWonN(5, source.copy(), False)
        print()
        # ---------------------------------------------
        # build summariesAvgN if it does not exist
        f_type = 'summariesAvgN'
        san = SummariesAvgN(self.sum_df.copy(), self.combineDir(f_type))
        san.buildSummariesAvgN(5, source.copy(), False)
        print()
        # ---------------------------------------------
        # build advancedStarterStats if it does not exist
        f_type = 'advancedStarterStats'
        ass = AdvancedStarterStats('qb', self.sdf, self.adf, self.combineDir(f_type))
        ass.buildAdvancedStarterStats(source.copy(), False)
        print()
        # ---------------------------------------------
        # build positionGroupSeasonAvgSnapPcts if it does not exist
        f_type = 'positionGroupSeasonAvgSnapPcts'
        pgsp = PositionGroupSeasonAvgSnapPcts(self.scdf, self.sdf, self.combineDir(f_type, True))
        pgsp.buildPositionGroupSeasonAvgSnapPcts(source.copy(), False)
        print()
        # ---------------------------------------------
        # build overUnders if it does not exist
        f_type = 'overUnders'
        ou = OverUnders(self.cd, self.tn, self.combineDir(f_type))
        ou.buildOverUnders(source.copy(), False)
        print()
        # ---------------------------------------------
        # join all features/create train
        self.joinAll(source, False)
        print()
        self.joinAll(source, True)
        print()
        # ---------------------------------------------
        # save models
        self.saveModels(isShort=False)
        print()
        self.saveModels(isShort=True)
        print()
        # # ---------------------------------------------
        # # use train to create all pred_targets predictions and join
        # self.createPredTrain()
        # print()
        return
    def new_main(self, week: int, year: int):
        # curr wy
        wy = str(week) + " | " + str(year)
        # curr starters
        if wy in self.sdf['wy'].values:
            sdf = self.sdf.loc[self.sdf['wy']==wy]
        else:
            s_dir = 'starters_' + str(year)[-2:] + '/'
            sdf = pd.read_csv("%s.csv" % (self.data_path + s_dir + "starters_w" + str(week)))
        # build new source/matchups for given week and year
        f_type = 'source'
        source = buildNewSource(week, year, self.combineDir(f_type))
        print()
        # ---------------------------------------------
        # declare list to store each feature dataframe
        df_list = []
        # ---------------------------------------------
        # build teamElos
        f_type = 'teamElos'
        elos = Elos(self.cd, self.combineDir(f_type))
        elos.setRawElos(None)
        elos.update(isNewYear=False)
        df_list.append((elos.createBoth(source.copy(), isNew=True), f_type))
        print()
        # ---------------------------------------------
        # build seasonInfo
        f_type = 'seasonInfo'
        df_list.append((buildNewSeasonInfo(source, self.cd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build pred_standings
        f_type = 'pred_standings'
        tb = TiebreakerAttributes(self.cd, self.sl, self.combineDir(f_type))
        tb.update()
        ps = PredStandings(self.sl, self.combineDir(f_type))
        ps.updatePredictions()
        df_list.append((ps.buildNewPredStandings(source.copy()), f_type))
        print()
        # ---------------------------------------------
        # build lastN_5
        f_type = 'avgN'
        df_list.append((buildNewAvgN(5, source, self.cd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build qbHalfGame
        f_type = 'qbHalfGame'
        df_list.append((buildNewQbHalfGame(source, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build lastWonN
        f_type = 'lastWonN'
        df_list.append((buildNewLastWonN(20, source, self.cd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build pointsAllowedN
        f_type = 'pointsAllowedN'
        df_list.append((buildNewPointsAllowedN(50, source, self.cd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build playerGrades
        f_type = 'playerGrades'
        pg_models = PgModels('qb', self.cd, self.combineDir(f_type))
        pg_models.update()
        pg = PlayerGrades('qb', self.combineDir(f_type))
        pg.update(isNewYear=False)
        df_list.append((pg.createBoth(source.copy(), sdf, isNew=True), f_type))
        print()
        # ---------------------------------------------
        # build seasonRankings
        f_type = 'seasonRankings'
        df_list.append((buildNewSeasonRankings(source, self.cd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build pointsN
        f_type = 'pointsN'
        df_list.append((buildNewPointsN(50, source, self.cd, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build vegaslines
        f_type = 'vegaslines'
        df_list.append((buildNewVegasLine(source, self.cd, self.tn, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build matchup info
        f_type = 'matchupInfo'
        df_list.append((buildNewMatchupInfo(source.copy(), self.cd, self.sl, self.combineDir(f_type)), f_type))
        print()
        # ---------------------------------------------
        # build coachElos
        f_type = 'coachElos'
        c_elos = CoachElos(self.cd, self.cdf, self.combineDir(f_type))
        c_elos.setRawElos(None)
        c_elos.update(isNewYear=False)
        df_list.append((c_elos.createBoth(source.copy(), isNew=True), f_type))
        print()
        # ---------------------------------------------
        # build playerGrades
        f_type = 'playerGrades'
        pg_models = PgModels('rb', self.cd, self.combineDir(f_type))
        pg_models.update()
        pg = PlayerGrades('rb', self.combineDir(f_type))
        pg.update(isNewYear=False)
        df_list.append((pg.createBoth(source.copy(), sdf, isNew=True), f_type))
        print()
        # ---------------------------------------------
        # build playerGrades
        f_type = 'playerGrades'
        pg_models = PgModels('wr', self.cd, self.combineDir(f_type))
        pg_models.update()
        pg = PlayerGrades('wr', self.combineDir(f_type))
        pg.update(isNewYear=False)
        df_list.append((pg.createBoth(source.copy(), sdf, isNew=True), f_type))
        print()
        # ---------------------------------------------
        # build starterInfo
        f_type = 'starterInfo'
        si = StarterInfo(self.combineDir(f_type))
        df_list.append((si.buildStarterInfo(source.copy(), isNew=True), f_type))
        print()
        # ---------------------------------------------
        # build starterMaddenRatings
        f_type = 'starterMaddenRatings'
        smr = StarterMaddenRatings(self.combineDir(f_type))
        df_list.append((smr.buildStarterMaddenRatings(source.copy(), sdf, self.rdf, True), f_type))
        print()
        # ---------------------------------------------
        # build lastGameStatsN
        f_type = 'lastGameStatsN'
        lgs = LastGameStatsN(self.cd.copy(), self.combineDir(f_type))
        df_list.append((lgs.buildLastGameStatsN(5, source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build lastOppWonN
        f_type = 'lastOppWonN'
        low = LastOppWonN(self.cd.copy(), self.combineDir(f_type))
        df_list.append((low.buildLastOppWonN(5, source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build summariesAvgN
        f_type = 'summariesAvgN'
        san = SummariesAvgN(self.sum_df.copy(), self.combineDir(f_type))
        df_list.append((san.buildSummariesAvgN(5, source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build advancedStarterStats
        f_type = 'advancedStarterStats'
        ass = AdvancedStarterStats('qb', sdf, self.adf, self.combineDir(f_type))
        df_list.append((ass.buildAdvancedStarterStats(source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build positionGroupSeasonAvgSnapPcts
        f_type = 'positionGroupSeasonAvgSnapPcts'
        pgsp = PositionGroupSeasonAvgSnapPcts(self.scdf, sdf, self.combineDir(f_type, True))
        df_list.append((pgsp.buildPositionGroupSeasonAvgSnapPcts(source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # build overUnders
        f_type = 'overUnders'
        ou = OverUnders(self.cd, self.tn, self.combineDir(f_type))
        df_list.append((ou.buildOverUnders(source.copy(), True), f_type))
        print()
        # ---------------------------------------------
        # merge test
        self.joinTest(source, df_list, False)
        print()
        self.joinTest(source, df_list, True)
        print()
        # ---------------------------------------------
        # check test has same features as train.csv
        self.setTest()
        self.setTrain()
        if self.train.shape[1] != self.test.shape[1]:
            print('Train shape: ' + str(self.train.shape[1]) + " != Test shape:" + str(self.test.shape[1]))
            return
        # ---------------------------------------------
        # make predictions
        self.setTest()
        self.predict(isShort=False)
        print()
        self.predict(isShort=True)
        print()
        # # ---------------------------------------------
        # # make pred_train predictions
        # self.predTrainPredict()
        # print()
        return
    def test_func(self):
        for pos in self.position_frames:
            df = self.position_frames[pos]
            print(pos)
            print(df.head())
        return
    # / END Main
    
####################################