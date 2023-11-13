import pandas as pd
import numpy as np
from functools import partial
import os
import regex as re
import multiprocessing
from ordered_set import OrderedSet
import time

from sportsipy.nfl.boxscore import Boxscores
from sportsipy.nfl.boxscore import Boxscore
from sportsipy.nfl.teams import Teams
from sportsipy.nfl.roster import Player

pd.options.mode.chained_assignment = None

STARTERS_PATH = "../starters/"

POSITIONS = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P', 'LS']

def shorten():

    df = pd.read_csv("%s.csv" % "gameData")

    start = df.loc[df['wy'].str.contains('1994')].index.values[0]

    df = df.loc[df.index>=start]

    df.to_csv("%s.csv" % "newGameData", index=False)
    
    return

def buildFantasyData():
    
    fns = [fn for fn in os.listdir('positionData/') if re.search(r"(QB|RB|WR|TE)[A-Z][a-z]+", fn)]
    
    df_list = []
    
    for fn in fns:
        df = pd.read_csv("positionData/" + fn)
        start = df.loc[df['wy']=='1 | 1994'].index.values[0]
        df_list.append(df.loc[df.index>=start])
        
    new_df = pd.concat(df_list)
    
    new_df.sort_values(by=['game_key', 'abbr'], inplace=True)
    
    new_df.to_csv("%s.csv" % "fantasyData", index=False)
    
    return

def drop2002SuperbowlFantasyData():
    key = '200301260rai'
    df = pd.read_csv("%s.csv" % "fantasyData")
    drops = df.loc[df['game_key']==key].index.values
    print(df.shape, len(drops))
    df.drop(drops, inplace=True)
    print(df.shape)
    df.to_csv("%s.csv" % "fantasyData", index=False)
    return

def drop2002SuperbowlWeek():
    key = '200301260rai'
    _dir = "positionData/"
    fns = [fn for fn in os.listdir(_dir) if '.csv' in fn]
    for fn in fns:
        print(fn)
        df = pd.read_csv(_dir + fn)
        drops = df.loc[df['game_key']==key].index.values
        df.drop(drops, inplace=True)
        df.to_csv(_dir + fn, index=False)
    return

def test():
    # BAD position/fantasy data key = 200301260rai
    key = '200301260rai'
    df = pd.read_csv("%s.csv" % "fantasyData")
    cd = pd.read_csv("%s.csv" % "gameData")
    df = df.loc[df['game_key']==key]
    cd = cd.loc[cd['key']==key]
    print(df, cd)
    return

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

def addPoints(cd: pd.DataFrame):
    cols = ['p_id', 'position', 'key', 'abbr', 'wy']
    new_df = pd.DataFrame(columns=cols+['points'])
    for index, row in cd.iterrows():
        pid = row['p_id']
        wy = row['wy']
        position = row['position']
        key = row['game_key']
        abbr = row['abbr']
        stats = cd.loc[(cd['p_id']==pid)&(cd['wy']==wy)].squeeze()
        if position == 'QB':
            points = getQbPoints(stats)
        else:
            points = getSkillPoints(stats)
        new_df.loc[len(new_df.index)] = [
            pid, position, key,
            abbr, wy
        ] + [points]
    return new_df

def buildFantasyData(cd: pd.DataFrame):
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    cd_split = np.array_split(cd, num_partitions)
    df_list = []
    if __name__ == '__main__':
        pool = multiprocessing.Pool(num_cores)
        all_dfs = pd.concat(pool.map(addPoints, cd_split))
        df_list.append(all_dfs)
        pool.close()
        pool.join()
        new_df = pd.concat(df_list)
        positions = ['QB', 'RB', 'WR', 'TE']
        wys = list(OrderedSet(cd['wy'].values))
        df_list = []
        for wy in wys:
            for position in positions:
                temp_df: pd.DataFrame = new_df.loc[(new_df['position']==position)&(new_df['wy']==wy)]
                temp_df.sort_values(by=['points'], ascending=False, inplace=True)
                temp_df.reset_index(drop=True, inplace=True)
                temp_df['week_rank'] = temp_df.index
                df_list.append(temp_df)
        new_df = new_df.merge(pd.concat(df_list), on=list(new_df.columns), how='left')
        new_df.to_csv("%s.csv" % "fantasyData", index=False)
    return

def sortFantasyData():
    df = pd.read_csv("%s.csv" % "fantasyData")
    df.sort_values(by=['key', 'abbr'], inplace=True)
    df.to_csv("%s.csv" % "fantasyData", index=False)
    return

def buildAllSummaries():
    
    cd = pd.read_csv("%s.csv" % "gameData")
    cd = cd.loc[cd['wy']=='3 | 1994']
    
    columns = ['key', 'wy', 'home_abbr', 'home_1', 'home_2', 'home_3', 'home_4', 'home_ot',
           'away_abbr', 'away_1', 'away_2', 'away_3', 'away_4', 'away_ot']

    new_df = pd.DataFrame(columns=columns)
    
    for index, row in cd.iterrows():
        key = row['key']
        wy = row['wy']
        print(wy)
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        boxscore = Boxscore(key)
        summary = boxscore.summary
        print(summary)
        home_sum = '|'.join([str(i) for i in summary['home']])
        away_sum = '|'.join([str(i) for i in summary['away']])
        # home
        home_sum = home_sum.split("|")
        if len(home_sum) < 5:
            home_sum.append(0)
        if len(home_sum) > 5:
            home_sum.pop(len(home_sum)-2)
        home_sum = [int(s) for s in home_sum]
        # away
        away_sum = away_sum.split("|")
        if len(away_sum) < 5:
            away_sum.append(0)
        if len(away_sum) > 5:
            away_sum.pop(len(away_sum)-2)
        away_sum = [int(s) for s in away_sum]
        new_df.loc[len(new_df.index)-1] = [
            row['key'], row['wy'], home_abbr
        ] + home_sum + [away_abbr] + away_sum
    
    new_df.to_csv("%s.csv" % "summaries", index=False)
    
    return

# combine positionData pids + starters
def allPids():
    sdf = pd.read_csv("%s.csv" % (STARTERS_PATH + "allStarters"))
    df = pd.concat([pd.read_csv("positionData/" + fn) for fn in os.listdir("positionData/") if '.csv' in fn])
    new_df = pd.DataFrame(columns=['key', 'wy', 'abbr', 'pids'])
    for index, row in sdf.iterrows():
        abbr = row['abbr']
        wy = row['wy']
        print(wy)
        starters = [(s.split(":")[0], s.split(":")[1]) for s in (row['starters']).split('|')]
        others = df.loc[(df['abbr']==abbr)&(df['wy']==wy), ['p_id', 'position']].values
        others = list(zip(others[:, 0], others[:, 1]))
        dif = list(set(others).difference(set(starters)))
        all_pids = '|'.join([(s[0] + ':' + s[1]) for s in dif]) + '|' + row['starters']
        new_df.loc[len(new_df.index)] = [row['key'], wy, abbr, all_pids]
    new_df.to_csv("%s.csv" % "allPids", index=False)
    return

# get all position players for every season
def buildAllSeasonPlayers():
    sdf = pd.read_csv("%s.csv" % (STARTERS_PATH + "allStarters"))
    df = pd.concat([pd.read_csv("positionData/" + fn) for fn in os.listdir("positionData/") if '.csv' in fn])
    cols = [(pos.lower() + 's') for pos in POSITIONS]
    new_df = pd.DataFrame(columns=['year', 'abbr']+cols)
    first_wy = sdf['wy'].values[0]
    last_wy = sdf['wy'].values[-1]
    years = [i for i in range(int(first_wy.split(" | ")[1]), int(last_wy.split(" | ")[1])+1)]
    for year in years:
        print(year)
        abbrs = list(set(sdf.loc[sdf['wy'].str.contains(str(year)), 'abbr'].values))
        for abbr in abbrs:
            all_starters = '|'.join(sdf.loc[(sdf['wy'].str.contains(str(year)))&(sdf['abbr']==abbr), 'starters'].values)
            all_starters = list(set([(s.split(":")[0], s.split(":")[1]) for s in all_starters.split("|")]))
            others = df.loc[(df['wy'].str.contains(str(year)))&(df['abbr']==abbr), ['p_id', 'position']].values
            others = list(zip(others[:, 0], others[:, 1]))
            all_players = list(set(all_starters + others))
            vals = []
            for pos in POSITIONS:
                pos_players = [s[0] for s in all_players if s[1] == pos]
                vals.append('|'.join(pos_players))
            new_df.loc[len(new_df.index)] = [year, abbr] + vals
    new_df.to_csv("%s.csv" % "allSeasonPlayers", index=False)
    return

# remove playoffs from gameData to gameData_regOnly
def removePlayoffs_gameData():
    df = pd.read_csv("%s.csv" % "gameData")
    sl = pd.read_csv("%s.csv" % "seasonLength")
    df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
    df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
    df_list = []
    for year, weeks in sl[['year', 'weeks']].values:
        if year != 2023:
            temp_df: pd.DataFrame = df.loc[df['year']==year]
            temp_df = temp_df.loc[temp_df['week']<=weeks]
            df_list.append(temp_df)
    new_df = pd.concat(df_list)
    new_df.to_csv("%s.csv" % "gameData_regOnly", index=False)
    return

# test sportsipy Boxscore
def test_sportsipy():
    
    box_list, df_list = [], []
    box_list.append(Boxscores(1, 2023))

    for boxscores in box_list:
        for key in boxscores.games.keys():
            for item in boxscores.games[key]:
                l1 = list(item.values())
                boxscore = Boxscore(l1[0])
                try:
                    temp_df = boxscore.dataframe
                    print(boxscore.winning_name)
                    temp_df['winning_abbr'] = boxscore.winning_abbr
                    temp_df['home_abbr'] = boxscore.home_abbreviation.upper()
                    temp_df['away_abbr'] = boxscore.away_abbreviation.upper()
                    df_list.append(temp_df)
                except TypeError:
                    print('type error:', l1[0])
    
    return

# fix gameData wrong winners
def fixGameData():
    df = pd.read_csv("%s.csv" % "gameData")
    start = df.loc[df['wy']=='22 | 2022'].index.values[0]
    new_df: pd.DataFrame = df.loc[df.index>=start]
    wrong_cols = [
        'losing_abbr', 'losing_name', 'winner',
        'winning_abbr', 'winning_name', 'home_points',
        'away_points'
    ]
    new_df['winner'] = new_df['winner'].apply(lambda x: 'Away' if x == 'Home' else 'Home')
    for suffix in ['_abbr', '_name']:
        ls, ws = new_df.copy()['losing'+suffix], new_df.copy()['winning'+suffix]
        new_df['winning'+suffix] = ls
        new_df['losing'+suffix] = ws
    hps, aps = new_df.copy()['home_points'], new_df.copy()['away_points']
    new_df['home_points'] = aps
    new_df['away_points'] = hps
    for index, row in new_df.iterrows():
        df.loc[df.index==index] = row.values
    df.to_csv("%s.csv" % "gameData_fixed", index=False)
    return

# get summaries
def getSummaries():
    box_list, df_list = [], []
    box_list.append(Boxscores(1, 2023))
    for boxscores in box_list:
        for key in boxscores.games.keys():
            for item in boxscores.games[key]:
                l1 = list(item.values())
                boxscore = Boxscore(l1[0])
                print(boxscore.summary)
    return

# get advanced stats
def getAdvancedStats():
    df = pd.read_csv("%s.csv" % "allPids")
    pids = '|'.join(df['pids'].values)
    pids = list(set([p.split(":")[0] for p in pids.split("|")]))
    pids.sort()
    df_list = []
    for index, pid in enumerate(pids):
        print(index, len(pids))
        p = Player(pid)
        pdf = p.dataframe
        if pdf is None:
            print("Empty frame")
        df_list.append(pdf)
        time.sleep(2)
    new_df = pd.concat(df_list)
    new_df = new_df[
        ['player_id', 'season', 'team_abbreviation']+[col for col in new_df.columns if col not in ['player_id', 'season', 'team_abbreviation']]
    ]
    new_df.drop(columns=['birth_date', 'height', 'weight'], inplace=True)
    new_df.to_csv("%s.csv" % "advancedStats", index=False)
    return

# replace missing over_unders
def insertMissingOverUnders():
    df = pd.read_csv("%s.csv" % "gameData")
    cd = pd.read_csv("%s.csv" % "missingOverUnders")
    for index, row in cd.iterrows():
        idx = df.loc[(df['key']==row['key'])&(df['wy']==row['wy'])].index.values[0]
        df.at[idx, 'over_under'] = row['over_under']
    df.to_csv("%s.csv" % "gameData", index=False)
    return

##########################

# buildFantasyData()

# drop2002SuperbowlFantasyData()

# drop2002SuperbowlWeek()

# test()

# sortFantasyData()

# buildAllSummaries()

# allPids()

# buildAllSeasonPlayers()

# removePlayoffs_gameData()

# test_sportsipy()

# fixGameData()

# getSummaries()

# getAdvancedStats()

insertMissingOverUnders()

# -------------------------------

# POSITION_PATH = "positionData/"

# fns = [fn for fn in os.listdir(POSITION_PATH) if re.search(r"(QB|RB|WR|TE)[A-Z][a-z]+", fn)]
# cd = pd.concat([pd.read_csv(POSITION_PATH + fn) for fn in fns])

# buildFantasyData(cd)