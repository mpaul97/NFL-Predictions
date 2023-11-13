import pandas as pd
import numpy as np
import os
from ordered_set import OrderedSet

pd.options.mode.chained_assignment = None

POSITIONS = ['QB', 'RB', 'WR', 'TE']

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

def getFantasyData(week, year, _dir):
    # p_id,position,key,abbr,wy,points,week_rank
    _dir += 'positionData/'
    fan_df = pd.DataFrame(columns=['p_id', 'position', 'key', 'abbr', 'wy', 'points'])
    skill_list = []
    for pos in POSITIONS:
        fn = pos + 'Data_' + str(week) + '-' + str(year) + '.csv'
        df = pd.read_csv(_dir + fn)
        skill_list.append(df)
        for index, row in df.iterrows():
            pid = row['p_id']
            key = row['game_key']
            abbr = row['abbr']
            wy = row['wy']
            if pos == 'QB':
                points = getQbPoints(row)
            else:
                points = getSkillPoints(row)
            fan_df.loc[len(fan_df.index)] = [pid, pos, key, abbr, wy, points]
    # add week ranks
    df_list = []
    for position in POSITIONS:
        temp_df: pd.DataFrame = fan_df.loc[(fan_df['position']==position)&(fan_df['wy']==wy)]
        temp_df.sort_values(by=['points'], ascending=False, inplace=True)
        temp_df.reset_index(drop=True, inplace=True)
        temp_df['week_rank'] = temp_df.index
        df_list.append(temp_df)
    fan_df = fan_df.merge(pd.concat(df_list), on=list(fan_df.columns), how='left')
    fan_df.sort_values(by=['key', 'abbr'], inplace=True)
    skill_df = pd.concat(skill_list)
    skill_df.sort_values(by=['game_key', 'abbr'], inplace=True)
    return fan_df, skill_df

def concatFantasyAndSkillData(week, year, _dir):
    f_df, s_df = getFantasyData(week, year, _dir)
    f_data = pd.read_csv("%s.csv" % "fantasyData")
    f_data = pd.concat([f_data, f_df])
    f_data.to_csv("%s.csv" % "fantasyData", index=False)
    s_data = pd.read_csv("%s.csv" % "skillData")
    s_data = pd.concat([s_data, s_df])
    s_data.to_csv("%s.csv" % "skillData", index=False)
    return

def createHalfAndStandard():
    df = pd.read_csv("%s.csv" % "fantasyData")
    df.drop(columns=['week_rank'], inplace=True)
    df.columns = ['p_id', 'position', 'key', 'abbr', 'wy', 'points_ppr']
    points_std, points_half = [], []
    return

def expandFpData():
    df = pd.read_csv("%s.csv" % "fantasyData")
    df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
    df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
    df.to_csv("%s.csv" % "fantasyData_expanded", index=False)
    return

#####################################

# createHalfAndStandard()

expandFpData()

# # make week folder
# week = 22
# year = 2022
# _dir = str(week) + "-" + str(year)
# if _dir not in os.listdir("newWeekData/"):
#     os.mkdir("newWeekData/" + _dir)
# _dir = "newWeekData/" + _dir
# _dir += "/"

# concatFantasyAndSkillData(week, year, _dir)