import pandas as pd
import numpy as np
import os
import urllib.request
import regex as re
import requests
import json
from datetime import datetime, timedelta
import sys
sys.path.append("../../../../")

from api_keys.the_odds import API_KEY

API_HOST = 'https://api.the-odds-api.com'

def buildVegasLine(source: pd.DataFrame, cd: pd.DataFrame, tn: pd.DataFrame, _dir):
    if 'vegaslines.csv' in os.listdir(_dir):
        print('vegaslines.csv already built.')
        return
    print('Creating vegaslines...')
    new_df = pd.DataFrame(columns=list(source.columns)+['home_vegasline', 'away_vegasline', 'home_isVegasExpectedWinner'])
    for index, row in source.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        line = cd.loc[cd['key']==key, 'vegas_line'].values[0]
        if line != '' and not pd.isna(line) and line != 'Pick':
            arr = line.split(" ")
            value = float(arr[-1])
            arr.pop()
            name = ' '.join(arr)
            abbr = tn.loc[tn['name'].str.contains(name), 'abbr'].values[0]
            isHome = (abbr == home_abbr)
            home_line = value if isHome else (value*-1)
            away_line = home_line*-1
            homeExpectedWinner = 1 if home_line < 0 else 0
        else: # even odds
            home_line = 0
            away_line = 0
            homeExpectedWinner = 0
        new_df.loc[len(new_df.index)] = list(row.values) + [home_line, away_line, homeExpectedWinner]
    new_df.to_csv("%s.csv" % (_dir + "vegaslines"), index=False)
    return

def getVegasLines_theLines(source: pd.DataFrame, tn: pd.DataFrame): # scrape vegas lines from thelines.com
    week = source['wy'].values[0].split(' | ')[0]
    year = source['wy'].values[0].split(' | ')[1]
    
    temp_df = pd.DataFrame(columns=['home_abbr', 'away_abbr', 'home_vegasline', 'away_vegasline', 'home_isVegasExpectedWinner'])
    
    url = "https://www.thelines.com/nfl-week-" + week +"-odds-" + year + "/"

    print(url)

    header = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }

    r = requests.get(url, headers=header)

    tables = pd.read_html(r.text)

    df = tables[0]

    game_col = [col for col in df.columns if 'Game' in col][0]
    spread_col = [col for col in df.columns if 'Spread' in col][0]

    spreads = df[[game_col, spread_col]].values

    for line, spread in spreads:
        arr = line.split(' at ')
        data = []
        home_name = arr[-1]
        away_name = arr[0]
        home_abbr = tn.loc[tn['name'].str.contains(home_name), 'abbr'].values[0]
        away_abbr = tn.loc[tn['name'].str.contains(away_name), 'abbr'].values[0]
        spread = spread.split(' ')
        bet = float(spread[-1])
        winning_abbr = tn.loc[tn['name'].str.contains(spread[0]), 'abbr'].values[0]
        home_isExpectedWinner = 1 if winning_abbr == home_abbr else 0
        home_bet = bet if home_isExpectedWinner else -1*bet
        away_bet = home_bet*-1
        temp_df.loc[len(temp_df.index)] = [home_abbr, away_abbr, home_bet, away_bet, home_isExpectedWinner]
        
    source = source.merge(temp_df, on=['home_abbr', 'away_abbr'])
        
    return source

def buildNewVegasLine_v1():
    # new_df = pd.DataFrame(columns=list(source.columns)+['home_vegasline', 'away_vegasline', 'home_isVegasExpectedWinner'])
    # for index, row in source.iterrows():
    #     key = row['key']
    #     home_abbr = row['home_abbr']
    #     away_abbr = row['away_abbr']
    #     if row['wy'] in cd['wy'].values:
    #         line = cd.loc[cd['key']==key, 'vegas_line'].values[0]
    #         if line != '' and not pd.isna(line) and line != 'Pick':
    #             arr = line.split(" ")
    #             value = float(arr[-1])
    #             arr.pop()
    #             name = ' '.join(arr)
    #             abbr = tn.loc[tn['name'].str.contains(name), 'abbr'].values[0]
    #             isHome = (abbr == home_abbr)
    #             home_line = value if isHome else (value*-1)
    #             away_line = home_line*-1
    #             homeExpectedWinner = 1 if home_line < 0 else 0
    #         else: # even odds
    #             home_line = 0
    #             away_line = 0
    #             homeExpectedWinner = 0
    #     else:
    #         break
    #     new_df.loc[len(new_df.index)] = list(row.values) + [home_line, away_line, homeExpectedWinner]
        
    # wy = source['wy'].values[0]
    # fn = "newVegaslines_" + wy.replace(' | ','-')
    # try:
    #     new_df = getVegasLines_theLines(source, tn)
    # except Exception:
    #     new_df = pd.read_csv("%s.csv" % (_dir + fn))
    # new_df.to_csv("%s.csv" % (_dir + fn), index=False)
    # return new_df
    return

def cleanTheOddsResponse(data, source: pd.DataFrame, tn: pd.DataFrame):
    df = pd.DataFrame(columns=['home_abbr', 'away_abbr', 'home_point', 'away_point'])
    for l in data:
        try:
            bms = l['bookmakers'][0]['markets'][0]
            outcomes = bms['outcomes']
            home_team, away_team = l['home_team'], l['away_team']
            home_point = [o['point'] for o in outcomes if o['name']==home_team][0]
            away_point = [o['point'] for o in outcomes if o['name']==away_team][0]
            home_abbr = tn.loc[tn['name']==home_team, 'abbr'].values[0]
            away_abbr = tn.loc[tn['name']==away_team, 'abbr'].values[0]
        except IndexError:
            print("Vegas lines missing game, using mean.")
            continue
        df.loc[len(df.index)] = [home_abbr, away_abbr, home_point, away_point]
    source = source.merge(df, on=['home_abbr', 'away_abbr'])
    return source

def getVegasLines_theOdds(source: pd.DataFrame, tn: pd.DataFrame):
    params = {
        'apiKey': API_KEY,
        'regions': 'us,us2',
        'markets': 'spreads',
        'bookmakers': 'draftkings'
    }
    url = API_HOST + '/v4/sports/americanfootball_nfl/odds/'
    res = requests.get(url, params=params)
    # json.dump(res.json(), open('temp.json', 'w'))
    # data = json.load(open('temp.json', 'r'))
    return cleanTheOddsResponse(res.json(), source.copy(), tn)

def buildNewVegasLine(source: pd.DataFrame, cd: pd.DataFrame, tn: pd.DataFrame, _dir):
    wy = source['wy'].values[0]
    fn = "newVegaslines_" + wy.replace(' | ','-')
    if ("%s.csv" % fn) not in os.listdir(_dir):
        print('Creating new vegaslines...')
        df: pd.DataFrame = getVegasLines_theOdds(source, tn)
        if len(source.index) > len(df.index):
            missing_keys = list(set(source['key']).difference(df['key']))
            for key in missing_keys:
                source_vals = source.loc[source['key']==key].values[0]
                df.loc[len(df.index)] = list(source_vals) + [np.nan, np.nan]
        df.fillna(df.mean(), inplace=True)
        df['home_isVegasExpectedWinner'] = df.apply(lambda x: 1 if x['home_point'] < x['away_point'] else 0, axis=1)
        df.columns = list(source.columns)+['home_vegasline', 'away_vegasline', 'home_isVegasExpectedWinner']
        # save on updating
        df.to_csv("%s.csv" % (_dir + fn), index=False)
        return df
    print('Using existing lines.')
    return pd.read_csv("%s.csv" % (_dir + fn))

##############################

# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")
# source = pd.read_csv("%s.csv" % "../source/new_source")
# tn = pd.read_csv("%s.csv" % "../../../../teamNames/teamNames_line")

# # df = getVegasLines_theLines(source, tn)

# # print(df)

# buildNewVegasLine(source, cd, tn, './')
