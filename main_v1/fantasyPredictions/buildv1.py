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

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
except ModuleNotFoundError:
    print('No modules found.')

STR_COLS = ['key', 'abbr', 'p_id', 'wy', 'position']
POSITIONS = ['QB', 'RB', 'WR', 'TE']

NON_CONT_MODELS = ['log', 'forest']

TARGET_OLS_THRESHOLDS = [
    {'name': 'points', 'value': 0.2},
    {'name': 'week_rank', 'value': 0.2}
]

ALL_MODELS = {
    'points': {
        'linear': LinearRegression(n_jobs=-1),
        # 'forestReg': RandomForestRegressor(n_jobs=-1),
        # 'log': LogisticRegression(n_jobs=-1)
        # 'forest': RandomForestClassifier(n_jobs=-1)
    },
    'week_rank': {
        'linear': LinearRegression(n_jobs=-1),
        # 'log': LogisticRegression(n_jobs=-1)
    }
}

POSITION_ALL_MODELS = {
    'points': {
        'forestReg': RandomForestRegressor(n_jobs=-1),
        'log': LogisticRegression(max_iter=500, verbose=0, n_jobs=-1)
    }
}

# get feature dir
def combineDir(root, _type):
    return root + _type + '/'

# get fantasy points for target
def getQbPoints(row: pd.Series):
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

# get fantasy points for target
def getSkillPoints(row: pd.Series):
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

# add points - parallel helper
def addPoints(source: pd.DataFrame, cd: pd.DataFrame):
    new_df = pd.DataFrame(columns=list(source.columns)+['points'])
    for index, row in source.iterrows():
        pid = row['p_id']
        wy = row['wy']
        position = row['position']
        stats = cd.loc[(cd['p_id']==pid)&(cd['wy']==wy)].squeeze()
        if position == 'QB':
            points = getQbPoints(stats)
        else:
            points = getSkillPoints(stats)
        new_df.loc[len(new_df.index)] = list(row.values) + [points]
    return new_df

# build target
def buildTarget(source: pd.DataFrame, cd: pd.DataFrame, _dir):
    if 'target.csv' in os.listdir(_dir):
        print('target.csv already created.')
        return
    print('Creating target...')
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    source_split = np.array_split(source, num_partitions)
    df_list = []
    if __name__ == 'fantasyPredictions.build':
        pool = multiprocessing.Pool(num_cores)
        all_dfs = pd.concat(pool.map(partial(addPoints, cd=cd), source_split))
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
        new_df.to_csv("%s.csv" % (_dir + "target"), index=False)
    return

# join all train features
def joinAll(source: pd.DataFrame, f_dir, _dir):
    print('Joining...')
    new_df = source.copy()
    trainInfo = pd.DataFrame(columns=['num', 'name'])
    num = 1
    for fn in os.listdir(f_dir):
        try:
            if 'source' not in fn:
                if re.search(r".*[N]$", fn): # fn ends with capital N
                    filename = [f for f in os.listdir(combineDir(f_dir, fn)) if fn in f and '.csv' in f][0]
                    df = pd.read_csv(combineDir(f_dir, fn) + filename)
                else:
                    df = pd.read_csv("%s.csv" % (combineDir(f_dir, fn) + fn))
                new_df = new_df.merge(df, on=list(source.columns), how='left')
                trainInfo.loc[len(trainInfo.index)] = [num, fn]
                num += 1
        except FileNotFoundError:
            print(fn, 'not created.')
    trainInfo.to_csv("%s.csv" % (_dir + "trainInfo"), index=False)
    new_df.to_csv("%s.csv" % (_dir + "train"), index=False)
    return

# join all test features
def joinTest(source: pd.DataFrame, df_list: list, f_dir, _dir):
    print('Joining test...')
    # sort df_list _types to match features directory order
    _types = [t for _, t in df_list]
    fns: list = os.listdir(f_dir)
    [fns.remove(fn) for fn in fns if fn not in _types]
    _all = [(fns.index(f), f, df) for df, f in df_list]
    _all.sort(key=lambda x: x[0])
    new_df = source.copy()
    for _, _, df in _all:
        new_df = new_df.merge(df, on=list(source.columns), how='left')
    new_df.to_csv("%s.csv" % (_dir + "test"), index=False)
    return new_df

# get ols stats / drops
def getOlsDrops(X, y, OLS_THRESHOLD, _dir):
    
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

# create/save models
def saveModels(showPreds, _dir):
    
    # clear old models
    [os.remove(_dir + 'models/' + fn) for fn in os.listdir(_dir + 'models/')]
    
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    
    t_cols = list(target.columns[len(STR_COLS):])
    
    drops_df = pd.DataFrame()
    
    for col in t_cols:
        threshold = [x['value'] for x in TARGET_OLS_THRESHOLDS if x['name']==col][0]
        data = train.merge(target[STR_COLS+[col]], on=STR_COLS, how='left')
        # if col == 'points': # remove outliers
        #     print("Data Shape before outliers:", data.shape)
        #     data: pd.DataFrame = data.loc[data[col].between(5, 40)]
        #     print("Data Shape after outliers:", data.shape)
        X = data.drop(columns=STR_COLS+[col])
        y = data[col]
        drops = getOlsDrops(X, y, threshold, "./")
        drops_df[col] = [','.join(drops)]
        print(col, len(drops))
        X = X.drop(drops, axis=1)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        joblib.dump(scaler, open((_dir + 'models/' + col + '-scaler.sav'), 'wb'))
        models = ALL_MODELS[col]
        for name in models:
            model = models[name]
            if name == 'log' or name == 'forest':
                y_train = y_train.round(0)
                y_test = y_test.round(0)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(name + '_accuracy - ' + col + ':', acc)
            joblib.dump(model, open((_dir + 'models/' + col + '_' + name + '.sav'), 'wb'))
            if showPreds:
                preds = model.predict(X_test)
                for i in range(10):
                    p = str(int(preds[i]))
                    e = str(y_test.values[i])
                    print('P: ' + p + ' <=> E: ' + e)
        
    print()
    drops_df.to_csv("%s.csv" % (_dir + "drops"), index=False)
    print('Drops created.')
                
    print('Models saved.')
    
    return

# create/save models - position
def saveModels_positions(_dir):
    
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    
    # t_cols = list(target.columns[len(STR_COLS):])
    t_cols = ['points']
    
    drops_df = pd.DataFrame()
    
    for position in POSITIONS:
        pos_train: pd.DataFrame = train.loc[train['position']==position]
        pos_target: pd.DataFrame = target.loc[target['position']==position]
        for col in t_cols:
            threshold = [x['value'] for x in TARGET_OLS_THRESHOLDS if x['name']==col][0]
            data = pos_train.merge(pos_target[STR_COLS+[col]], on=STR_COLS, how='left')
            X = data.drop(columns=STR_COLS+[col])
            y = data[col]
            drops = getOlsDrops(X, y, threshold, "./")
            drops_df[position + "_" + col] = [','.join(drops)]
            X = X.drop(drops, axis=1)
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            joblib.dump(scaler, open((_dir + 'models/' + position + '_' + col + '-scaler.sav'), 'wb'))
            models = POSITION_ALL_MODELS[col]
            for name in models:
                model = models[name]
                if name in NON_CONT_MODELS:
                    y_train = y_train.round(0)
                    y_test = y_test.round(0)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                print(position + '_' + name + '_accuracy - ' + col + ':', acc)
                joblib.dump(model, open((_dir + 'models/' + position + '_' + col + '_' + name + '.sav'), 'wb'))
          
    drops_df.to_csv("%s.csv" % (_dir + "drops_positions"), index=False)
    print("Position drops created.")
            
    print('Individual position models saved.')
            
    return

# load models and create predictions
def predict(test: pd.DataFrame, _dir):
    print('Predicting...')
    m_dir = _dir + 'models/'
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    t_cols = list(target.columns[len(STR_COLS):])
    X = test.drop(columns=STR_COLS)
    drops_df = pd.read_csv("%s.csv" % (_dir + "drops"))
    all_names = []
    for t_name in t_cols:
        print('Model Name:', t_name)
        if type(drops_df[t_name].values[0]) is str:
            drops = drops_df[t_name].values[0].split(',')
            X1 = X.drop(drops, axis=1)
        else:
            X1 = X.copy()
        scaler: StandardScaler = joblib.load(open((m_dir + t_name + '-scaler.sav'), 'rb'))
        X_scaled = scaler.transform(X1)
        fns = [fn for fn in os.listdir(m_dir) if t_name in fn and 'scaler' not in fn and len(fn.split("_")) < 3]
        for fn in fns:
            modelName = fn.replace('.sav','')
            print('Model name:', modelName)
            model = joblib.load(open((m_dir + fn), 'rb'))
            preds = model.predict(X_scaled)
            test[modelName] = preds
            all_names.append(modelName)
    test = test.round(2)
    test = test[STR_COLS+all_names]
    sort_col = [n for n in all_names if 'points' in n][0]
    test.sort_values(by=[sort_col], ascending=False, inplace=True)
    test.to_csv("%s.csv" % (_dir + "predictions"), index=False)
    # save each position predictions
    for pos in POSITIONS:
        temp_df = test.loc[test['position']==pos]
        temp_df.sort_values(by=[sort_col], ascending=False, inplace=True)
        temp_df.to_csv("%s.csv" % (_dir + "predictions_" + pos), index=False)
    return

# load models and create predictions - positions
def predict_positions(test: pd.DataFrame, _dir):
    print('Predicting individual positions...')
    m_dir = _dir + 'models/'
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    # t_cols = list(target.columns[len(STR_COLS):])
    t_cols = ['points']
    drops_df = pd.read_csv("%s.csv" % (_dir + "drops_positions"))
    all_preds = { pos: [] for pos in POSITIONS }
    for position in POSITIONS:
        X = test.loc[test['position']==position]
        X_copy = X[STR_COLS]
        X = X.drop(columns=STR_COLS)
        for t_name in t_cols:
            print(f"Model params: {position} - {t_name}")
            drops_col = position + "_" + t_name
            # if type(drops_df[drops_col].values[0]) is str:
            drops = drops_df[drops_col].values[0].split(',')
            X1 = X.drop(drops, axis=1)
            # else:
            #     X1 = X.copy()
            scaler: StandardScaler = joblib.load(open((m_dir + position + '_' + t_name + '-scaler.sav'), 'rb'))
            X_scaled = scaler.transform(X1)
            fns = [fn for fn in os.listdir(m_dir) if t_name in fn and 'scaler' not in fn and position in fn]
            for fn in fns:
                modelName = fn.replace('.sav','')
                print('Model name:', modelName)
                model = joblib.load(open((m_dir + fn), 'rb'))
                preds = model.predict(X_scaled)
                X_copy[modelName] = preds
        all_preds[position].append(X_copy)
    for position in POSITIONS:
        frames = all_preds[position]
        df = reduce(lambda x, y: pd.merge(x, y, on=STR_COLS), frames)
        sort_col = [col for col in df.columns if 'points' in col and position in col][0]
        df = df.round(2)
        fdf = pd.read_csv("%s.csv" % (_dir + 'predictions_' + position))
        fdf = fdf.merge(df, on=STR_COLS)
        fdf.sort_values(by=[sort_col], ascending=False, inplace=True)
        fdf.to_csv("%s.csv" % (_dir + 'predictions_' + position), index=False)
    return

# --------------------
# main functions 
# --------------------

# build features, train, and target
def main(all_paths, _dir):
    
    # paths
    DATA_PATH = all_paths['dp']
    POSITION_PATH = all_paths['pp']
    STARTERS_PATH = all_paths['sp']
    MADDEN_PATH = all_paths['mrp']
    
    # load skill (all skill player data - rush_yards, completed_passes, ...) data
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "skillData"))
    
    # load fantasy data (points and week_rank)
    fd = pd.read_csv("%s.csv" % (DATA_PATH + "fantasyData"))
    
    # all player data - from 1978 to present
    fns = [fn for fn in os.listdir(POSITION_PATH) if re.search(r"(QB|RB|WR|TE)[A-Z][a-z]+", fn)]
    ocd = pd.concat([pd.read_csv(POSITION_PATH + fn) for fn in fns])
    
    # each position frame
    positions = ['QB', 'RB', 'WR', 'TE']
    pos_data = { pos: pd.read_csv("%s.csv" % (POSITION_PATH + pos + "Data")) for pos in positions }
    
    # game data from data
    gd = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))
    ogd = pd.read_csv("%s.csv" % (DATA_PATH + "oldGameData_78"))
    
    # season length from data
    sl = pd.read_csv("%s.csv" % (DATA_PATH + "seasonLength"))
    
    # madden ratings
    rdf = pd.read_csv("%s.csv" % (MADDEN_PATH + "playerRatings"))
    
    # starters
    sdf = pd.read_csv("%s.csv" % (STARTERS_PATH + "allStarters"))
    
    # root features directory
    f_dir = _dir + 'features/'
    
    # build source if it does not exist
    f_type = 'source'
    source = buildSource(cd, combineDir(f_dir, f_type))
    print()
    
    # build target if it does not exist
    buildTarget(source, cd, _dir)
    print()
    
    # target data for ols stuff
    target = pd.read_csv("%s.csv" % (_dir + "target"))
    
    # build seasonAvg
    f_type = 'seasonAvg'
    buildSeasonAvg(source.copy(), fd, combineDir(f_dir, f_type))
    print()
    
    # build encodedPosition
    f_type = 'encodedPosition'
    buildEncodedPosition(source.copy(), combineDir(f_dir, f_type))
    print()
    
    # build maxWeekRank
    f_type = 'maxWeekRank'
    buildMaxWeekRank(source.copy(), fd, combineDir(f_dir, f_type))
    print()
    
    # build pointsN
    f_type = 'pointsN'
    buildPointsN(5, source.copy(), combineDir(f_dir, f_type))
    print()
    
    # build careerAvg
    f_type = 'careerAvg'
    buildCareerAvg(source.copy(), combineDir(f_dir, f_type))
    print()
    
    # build allowedPointsN
    f_type = 'allowedPointsN'
    buildAllowedPointsN(5, source.copy(), combineDir(f_dir, f_type))
    print()
    
    # build season rankings
    f_type = 'seasonRankings'
    buildSeasonRankings(source.copy(), combineDir(f_dir, f_type))
    print()
    
    # build isPrimary
    f_type = 'isPrimary'
    ip = IsPrimary(combineDir(f_dir, f_type))
    ip.buildIsPrimary(source.copy(), ocd)
    print()
    
    # build maddenRatings
    f_type = 'maddenRatings'
    mr = MaddenRatings(combineDir(f_dir, f_type))
    mr.buildMaddenRatings(source.copy(), rdf, False)
    print()
    
    # build isStarter
    f_type = 'isStarter'
    iss = IsStarter(combineDir(f_dir, f_type))
    iss.buildIsStarter(source.copy(), sdf, False)
    print()
    
    # lastStatsN
    f_type = "lastStatsN"
    lsn = LastStatsN(pos_data, combineDir(f_dir, f_type))
    lsn.buildLastStatsN(10, source.copy(), False)
    print()
    
    # build avgStatsN if does not exist
    f_type = 'avgStatsN'
    asn = AvgStatsN(pos_data, combineDir(f_dir, f_type))
    asn.buildAvgStatsN_parallel(5, source.copy())
    print()
    
    # join all features/create train
    joinAll(source, f_dir, _dir)
    print()
    
    # save models
    saveModels(False, _dir)
    print()
    
    # save models positions
    saveModels_positions(_dir)
    print()
    
    return

# make test file for new week
def new_main(week, year, all_paths, _dir):
    
    # paths
    DATA_PATH = all_paths['dp']
    POSITION_PATH = all_paths['pp']
    STARTERS_PATH = all_paths['sp']
    MADDEN_PATH = all_paths['mrp']
    
    # new wy
    wy = str(week) + ' | ' + str(year)
    
    # load skill (all skill player data - rush_yards, completed_passes, ...) data
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "skillData"))
    
    # load fantasy data (points and week_rank)
    fd = pd.read_csv("%s.csv" % (DATA_PATH + "fantasyData"))
    
    # all player data - from 1978 to present
    fns = [fn for fn in os.listdir(POSITION_PATH) if re.search(r"(QB|RB|WR|TE)[A-Z][a-z]+", fn)]
    ocd = pd.concat([pd.read_csv(POSITION_PATH + fn) for fn in fns])
    
    # each position frame
    positions = ['QB', 'RB', 'WR', 'TE']
    pos_data = { pos: pd.read_csv("%s.csv" % (POSITION_PATH + pos + "Data")) for pos in positions }
    
    # game data from data
    gd = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))
    ogd = pd.read_csv("%s.csv" % (DATA_PATH + "oldGameData_78"))
    
    # all starters
    all_starters = pd.read_csv("%s.csv" % (STARTERS_PATH + "allStarters"))
    
    # curr starters
    if wy in all_starters['wy'].values:
        sdf = all_starters.loc[all_starters['wy']==wy]
    else:
        s_dir = 'starters_' + str(year)[-2:] + '/'
        sdf = pd.read_csv("%s.csv" % (DATA_PATH + s_dir + "starters_w" + str(week)))
        
    # madden ratings
    rdf = pd.read_csv("%s.csv" % (MADDEN_PATH + "playerRatings"))
    
    # root features directory
    f_dir = _dir + 'features/'
    
    # build new source/matchups for given week and year
    f_type = 'source'
    source = buildNewSource(week, year, fd, sdf, combineDir(f_dir, f_type))
    print()
    
    # declare list to store each feature dataframe
    df_list = []
    
    # build seasonAvg
    f_type = 'seasonAvg'
    df_list.append((buildNewSeasonAvg(source.copy(), fd, combineDir(f_dir, f_type)), f_type))
    print()
    
    # build encodedPosition
    f_type = 'encodedPosition'
    df_list.append((buildNewEncodedPosition(source.copy(), combineDir(f_dir, f_type)), f_type))
    print()
    
    # build maxWeekRank
    f_type = 'maxWeekRank'
    df_list.append((buildNewMaxWeekRank(source.copy(), combineDir(f_dir, f_type)), f_type))
    print()
    
    # build pointsN
    f_type = 'pointsN'
    df_list.append((buildNewPointsN(5, source.copy(), fd, combineDir(f_dir, f_type)), f_type))
    print()
    
    # build careerAvg
    f_type = 'careerAvg'
    df_list.append((buildNewCareerAvg(source.copy(), fd, combineDir(f_dir, f_type)), f_type))
    print()
    
    # build allowedPointsN
    f_type = 'allowedPointsN'
    df_list.append((buildNewAllowedPointsN(5, source.copy(), fd, gd, combineDir(f_dir, f_type)), f_type))
    print()
    
    # build seasonRankings
    f_type = 'seasonRankings'
    df_list.append((buildNewSeasonRankings(source.copy(), gd, combineDir(f_dir, f_type)), f_type))
    print()
    
    # build isPrimary
    f_type = 'isPrimary'
    ip = IsPrimary(combineDir(f_dir, f_type))
    df_list.append((ip.buildNewIsPrimary(source.copy(), ocd), f_type))
    print()
    
    # build maddenRatings
    f_type = 'maddenRatings'
    mr = MaddenRatings(combineDir(f_dir, f_type))
    df_list.append((mr.buildMaddenRatings(source.copy(), rdf, True), f_type))
    print()
    
    # build isStarter
    f_type = 'isStarter'
    iss = IsStarter(combineDir(f_dir, f_type))
    df_list.append((iss.buildIsStarter(source.copy(), sdf, True), f_type))
    print()
    
    # build lastStatsN
    f_type = "lastStatsN"
    lsn = LastStatsN(pos_data, combineDir(f_dir, f_type))
    df_list.append((lsn.buildLastStatsN(10, source.copy(), True), f_type))
    print()
    
    # build avgStatsN
    f_type = 'avgStatsN'
    asn = AvgStatsN(pos_data, combineDir(f_dir, f_type))
    df_list.append((asn.buildAvgStatsN(5, source.copy(), True), f_type))
    print()
    
    # merge test
    test = joinTest(source, df_list, f_dir, _dir)
    print()
    
    # check test has same features as train.csv
    train = pd.read_csv("%s.csv" % (_dir + "train"))
    if train.shape[1] != test.shape[1]:
        print('Train shape: ' + str(train.shape[1]) + " != Test shape:" + str(test.shape[1]))
        return
    
    # test = pd.read_csv("%s.csv" % (_dir + "test"))
    
    # make predictions
    predict(test.copy(), _dir)
    print()
    
    # make position predictions
    predict_positions(test.copy(), _dir)
    print()
    
    return

#####################

# saveModels(False, "./")

# predict(pd.read_csv("%s.csv" % "test"), "./")

# saveModels_positions("./")

# predict_positions(pd.read_csv("%s.csv" % "test"), './')

