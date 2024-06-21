import pandas as pd
import numpy as np
import os
import urllib.request
import regex as re
import requests
import json
from datetime import datetime, timedelta
import sys
sys.path.append("../../../")

from api_keys.the_odds import API_KEY

API_HOST = 'https://api.the-odds-api.com'

class VegasLines:
    def __init__(self, df: pd.DataFrame, tn: pd.DataFrame, _dir: str):
        self.df = df
        self.tn = tn
        self._dir = _dir
        self.str_cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        self.cols = ['home_vegasline', 'away_vegasline', 'home_isVegasExpectedWinner']
        return
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % (self._dir + name), index=False)
        return
    def lambda_func(self, row: pd.Series):
        home_abbr = row['home_abbr']
        line = row['vegas_line']
        if line != '' and not pd.isna(line) and line != 'Pick':
            arr: list[str] = line.split(" ")
            value = float(arr[-1])
            arr.pop()
            name = ' '.join(arr)
            abbr = self.tn.loc[self.tn['name'].str.contains(name), 'abbr'].values[0]
            isHome = (abbr == home_abbr)
            home_line = value if isHome else (value*-1)
            away_line = home_line*-1
            homeExpectedWinner = 1 if home_line < 0 else 0
        else: # even odds
            home_line = 0
            away_line = 0
            homeExpectedWinner = 0
        return home_line, away_line, homeExpectedWinner
    def buildVegasLine(self, source: pd.DataFrame):
        if 'vegaslines.csv' in os.listdir(self._dir):
            print('vegaslines.csv already built.')
            return
        print('Creating vegaslines...')
        source = source.merge(self.df[['key', 'vegas_line']], on=['key'])
        source[self.cols] = source.apply(lambda row: self.lambda_func(row), result_type='expand', axis=1)
        source = source[self.str_cols+self.cols]
        source['home_isVegasExpectedWinner'] = source['home_isVegasExpectedWinner'].astype(int)
        self.save_frame(source, "vegaslines")
        return
    def cleanTheOddsResponse(self, data, source: pd.DataFrame):
        df = pd.DataFrame(columns=['home_abbr', 'away_abbr', 'home_point', 'away_point'])
        for l in data:
            try:
                bms = l['bookmakers'][0]['markets'][0]
                outcomes = bms['outcomes']
                home_team, away_team = l['home_team'], l['away_team']
                home_point = [o['point'] for o in outcomes if o['name']==home_team][0]
                away_point = [o['point'] for o in outcomes if o['name']==away_team][0]
                home_abbr = self.tn.loc[self.tn['name']==home_team, 'abbr'].values[0]
                away_abbr = self.tn.loc[self.tn['name']==away_team, 'abbr'].values[0]
            except IndexError:
                print("Vegas lines missing game, using mean.")
                continue
            df.loc[len(df.index)] = [home_abbr, away_abbr, home_point, away_point]
        source = source.merge(df, on=['home_abbr', 'away_abbr'])
        return source
    def getVegasLines_theOdds(self, source: pd.DataFrame):
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
        return self.cleanTheOddsResponse(res.json(), source.copy())
    def buildNewVegasLine(self, source: pd.DataFrame):
        wy = source['wy'].values[0]
        fn = "vegaslines_new_" + wy.replace(' | ','-')
        if ("%s.csv" % fn) not in os.listdir(self._dir):
            print('Creating new vegaslines...')
            df: pd.DataFrame = self.getVegasLines_theOdds(source)
            if len(source.index) > len(df.index):
                missing_keys = list(set(source['key']).difference(df['key']))
                for key in missing_keys:
                    source_vals = source.loc[source['key']==key].values[0]
                    df.loc[len(df.index)] = list(source_vals) + [np.nan, np.nan]
            df.fillna(df.mean(), inplace=True)
            df['home_isVegasExpectedWinner'] = df.apply(lambda x: 1 if x['home_point'] < x['away_point'] else 0, axis=1)
            df.columns = list(source.columns)+['home_vegasline', 'away_vegasline', 'home_isVegasExpectedWinner']
            # save on updating
            self.save_frame(df, fn)
            return df
        print('Using existing vegaslines.')
        return pd.read_csv("%s.csv" % (self._dir + fn))
    def build(self, source: pd.DataFrame, isNew: bool = False):
        if isNew:
            return self.buildNewVegasLine(source)
        self.buildVegasLine(source)
        return
    
# end VegasLines

# vl = VegasLines(
#     df=pd.read_csv("%s.csv" % "../../../data/gameData"),
#     tn=pd.read_csv("%s.csv" % "../../../teamNames/teamNames_line"),
#     _dir="data/"
# )

# # vl.buildVegasLine(pd.read_csv("%s.csv" % "data/source"))
# vl.buildNewVegasLine(pd.read_csv("%s.csv" % "data/source_new"))