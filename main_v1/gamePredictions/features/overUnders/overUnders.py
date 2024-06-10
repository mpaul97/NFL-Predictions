import pandas as pd
import numpy as np
import os
import requests

pd.options.mode.chained_assignment = None

class OverUnders:
    def __init__(self, df: pd.DataFrame, tn: pd.DataFrame, _dir: str):
        self.df = df
        self.tn = tn
        self._dir = _dir
        # info
        self.API_KEY = '1774f96665d93ee33983b187ec18673d'
        self.API_HOST = 'https://api.the-odds-api.com'
        return
    def cleanTheOddsResponse(self, data, source: pd.DataFrame):
        df = pd.DataFrame(columns=['home_abbr', 'away_abbr', 'over_under'])
        for l in data:
            try:
                bms = l['bookmakers'][0]['markets'][0]
                outcomes = bms['outcomes']
                home_team, away_team = l['home_team'], l['away_team']
                over_under = float(outcomes[0]['point'])
                home_abbr = self.tn.loc[self.tn['name']==home_team, 'abbr'].values[0]
                away_abbr = self.tn.loc[self.tn['name']==away_team, 'abbr'].values[0]
            except IndexError:
                print("Vegas lines missing game, using mean.")
                continue
            df.loc[len(df.index)] = [home_abbr, away_abbr, over_under]
        source = source.merge(df, on=['home_abbr', 'away_abbr'])
        return source
    def getOverUnders_theOdds(self, source: pd.DataFrame):
        params = {
            'apiKey': self.API_KEY,
            'regions': 'us,us2',
            'markets': 'totals',
            'bookmakers': 'draftkings'
        }
        url = self.API_HOST + '/v4/sports/americanfootball_nfl/odds/'
        res = requests.get(url, params=params)
        return self.cleanTheOddsResponse(res.json(), source.copy())
    def buildOverUnders(self, source: pd.DataFrame, isNew: bool):
        wy = source['wy'].values[0]
        fn = "newOverUnders_" + wy.replace(' | ','-') if isNew else "overUnders"
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        if not isNew or wy in self.df['wy'].values:
            df = self.df[['key', 'over_under']]
            df['over_under'] = df['over_under'].apply(lambda x: float(x.split(" ")[0]))
            new_df = source.merge(df, on=['key'])
        else: # not new or not in gameData
            if ("%s.csv" % fn) not in os.listdir(self._dir):
                print('Creating new overUnders...')
                new_df = self.getOverUnders_theOdds(source)
                self.saveFrame(new_df, (self._dir + fn))
                return new_df
            else:
                print('Using existing overUnders.')
                return pd.read_csv("%s.csv" % (self._dir + fn))
        self.saveFrame(new_df, (self._dir + fn))
        return new_df
    def saveFrame(self, df: pd.DataFrame, name):
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
    
# END / OverUnders

######################

# df = pd.read_csv("%s.csv" % "../../../../data/gameData")
# tn = pd.read_csv("%s.csv" % "../../../../teamNames/teamNames_line")
# ou = OverUnders(df, tn, "./")

# source = pd.read_csv("%s.csv" % "../source/source")
# ou.buildOverUnders(source, False)

# source = pd.read_csv("%s.csv" % "../source/new_source")
# ou.buildOverUnders(source, True)