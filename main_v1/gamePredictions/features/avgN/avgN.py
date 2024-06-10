import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# get ols stats
def getOlsStats(X, y):
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    
    pdf: pd.Series = results.pvalues.sort_values()

    return pdf.to_frame()

def createDrops(df: pd.DataFrame, target: pd.DataFrame, threshold, _dir):
    
    X = df.drop(columns=['key', 'wy', 'home_abbr', 'away_abbr'])
    y = target['home_won']
    
    ols = getOlsStats(X, y)
    
    p_drops = []
    
    for index, row in ols.iterrows():
        val = row[0]
        if val > threshold and index != 'const':
            p_drops.append(index)
    
    new_df = pd.DataFrame()
    new_df['drops'] = p_drops
    
    new_df.to_csv("%s.csv" % (_dir + "drops"), index=False)
    
    # X = X.drop(columns=p_drops)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    
    # acc = model.score(X_test, y_test)
    
    # print('Accuracy:', acc)
    
    return

####################################

def replaceCols(df: pd.DataFrame, isHome, n):
    if isHome:
        col_names = list(df.columns)
        for name in col_names:
            new_col_name = name.replace('home_', str(n) + '_').replace('away_', str(n) + '_opp_')
            df = df.rename(columns={name: new_col_name})
    else:
        col_names = list(df.columns)
        for name in col_names:
            new_col_name = name.replace('away_', str(n) + '_').replace('home_', str(n) + '_opp_')
            df = df.rename(columns={name: new_col_name})
    return df

def getStats(stats: pd.DataFrame, cd: pd.DataFrame, start, abbr, n):
    if stats.empty:
        if start != -1:
            statsH = cd.loc[cd.index<start].tail(20)
        else:
            statsH = cd.tail(20)
        statsH = replaceCols(statsH, True, n)
        if start != -1:
            statsA = cd.loc[cd.index<start].tail(20)
        else:
            statsH = cd.tail(20)
        statsA = replaceCols(statsA, False, n)
        stats = pd.concat([statsH, statsA])
    else:
        homeStats = stats.loc[stats['home_abbr']==abbr]
        homeStats = replaceCols(homeStats, True, n)
        awayStats = stats.loc[stats['away_abbr']==abbr]
        awayStats = replaceCols(awayStats, False, n)
        stats = pd.concat([homeStats, awayStats])
    num = len(stats.index)
    stats = stats.sum(numeric_only=True).to_frame().transpose()
    stats = stats.apply(lambda x: x/num)
    return stats

def buildAvgN(n, source: pd.DataFrame, cd: pd.DataFrame, target: pd.DataFrame, drops_threshold, _dir):
    
    if 'avgN_' + str(n) + '.csv' in os.listdir(_dir):
        print('avgN_' + str(n) + '.csv already created.')
        new_df = pd.read_csv("%s.csv" % (_dir + "avgN_" + str(n)))
        drops = pd.read_csv("%s.csv" % (_dir + "drops"))
        if drops['drops'].values[0] in new_df.columns:
            print('Dropping columns...')
            new_df.drop(columns=drops['drops'].values, inplace=True)
        new_df.to_csv("%s.csv" % (_dir + 'avgN_' + str(n)), index=False)
        return
    
    drop_cols = [
        'attendance', 'stadium_id', 'lineHit', 
        'month', 'ouHit', 'time', 'surface'
    ]
    
    cd.drop(columns=drop_cols, inplace=True)
    
    stats_list = []
    
    print('Creating avgN_' + str(n) + '...')
    
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        start = cd.loc[cd['wy']==wy].index.values[0]
        home_stats = cd.loc[
            ((cd['home_abbr']==home_abbr)|(cd['away_abbr']==home_abbr))&
            (cd.index<start)
        ].tail(n)
        away_stats = cd.loc[
            ((cd['home_abbr']==away_abbr)|(cd['away_abbr']==away_abbr))&
            (cd.index<start)
        ].tail(n)
        home_stats = getStats(home_stats, cd, start, home_abbr, n)
        away_stats = getStats(away_stats, cd, start, away_abbr, n)
        home_stats.columns = ['home_' + col for col in home_stats.columns]
        away_stats.columns = ['away_' + col for col in away_stats.columns]
        all_stats = pd.concat([home_stats, away_stats], axis=1)
        all_stats.insert(0, 'key', key)
        stats_list.append(all_stats)
        
    new_df = pd.concat(stats_list)
    
    new_df = source.merge(new_df, on=['key'])

    print('Creating drops using OLS stats...')
    createDrops(new_df, target, drops_threshold, _dir)
    
    drops = pd.read_csv("%s.csv" % (_dir + "drops"))
    new_df.drop(columns=drops['drops'].values, inplace=True)
    
    new_df.to_csv("%s.csv" % (_dir + 'avgN_' + str(n)), index=False)
    
    return

def buildNewAvgN(n, source: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    print('Creating newAvgN_' + str(n) + '...')
    
    drop_cols = [
        'attendance', 'stadium_id', 'lineHit', 
        'month', 'ouHit', 'time', 'surface'
    ]
    
    cd.drop(columns=drop_cols, inplace=True)
    
    stats_list = []
    
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        try:
            start = cd.loc[cd['wy']==wy].index.values[0]
            home_stats = cd.loc[
                ((cd['home_abbr']==home_abbr)|(cd['away_abbr']==home_abbr))&
                (cd.index<start)
            ].tail(n)
            away_stats = cd.loc[
                ((cd['home_abbr']==away_abbr)|(cd['away_abbr']==away_abbr))&
                (cd.index<start)
            ].tail(n)
        except IndexError:
            start = -1
            home_stats = cd.loc[
                ((cd['home_abbr']==home_abbr)|(cd['away_abbr']==home_abbr))
            ].tail(n)
            away_stats = cd.loc[
                ((cd['home_abbr']==away_abbr)|(cd['away_abbr']==away_abbr))
            ].tail(n)
        home_stats = getStats(home_stats, cd, start, home_abbr, n)
        away_stats = getStats(away_stats, cd, start, away_abbr, n)
        home_stats.columns = ['home_' + col for col in home_stats.columns]
        away_stats.columns = ['away_' + col for col in away_stats.columns]
        all_stats = pd.concat([home_stats, away_stats], axis=1)
        all_stats.insert(0, 'key', key)
        stats_list.append(all_stats)
        
    new_df = pd.concat(stats_list)
    
    new_df = source.merge(new_df, on=['key'])
    
    drops = pd.read_csv("%s.csv" % (_dir + "drops"))
    new_df.drop(columns=drops['drops'].values, inplace=True)
    
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + 'newAvgN_' + str(n)), index=False)
    
    return new_df

########################

# source = pd.read_csv("%s.csv" % "../source/source")
# cd = pd.read_csv("%s.csv" % "../../../../data/oldGameData_78")
# target = pd.read_csv("%s.csv" % "../../target")

# buildAvgN(5, source, cd, target, 0.2, './')

# createDrops()

# source = pd.read_csv("%s.csv" % "../source/new_source")
# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")

# new_df = buildNewavgN(5, source, cd, './') # too many
# df = pd.read_csv("%s.csv" % "avgN_5")

# print(df.shape)