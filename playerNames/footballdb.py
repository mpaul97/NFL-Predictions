import pandas as pd
import numpy as np
import os
import requests
import urllib.request
import time

class FootballDb:
    def __init__(self, _dir):
        self._dir = _dir
        self.mr_dir = self._dir + "../maddenRatings/"
        return
    def getPosition(self, name: str, year: int, abbr: str):
        pos_df = pd.read_csv("%s.csv" % "positionsFinalPlayerInfo")
        last_name = ' '.join(name.split(" ")[1:])
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        }
        url = "https://www.footballdb.com/players/players.html?q=" + last_name.lower()
        res = requests.get(url, headers=headers)
        tables = pd.read_html(res.content.decode())
        try:
            df = tables[0]
            search_name = ' '.join(name.split(' ')[1:]) + ', ' + name.split(' ')[0]
            info = df.loc[(df['Player'].str.contains(search_name))&(df['Teams'].str.contains(str(year)))]
            info.reset_index(drop=True, inplace=True)
            if len(info.index) > 1:
                print(f"Arguments: {name}, {year}, {abbr}")
                print('-----------------------------')
                print(info)
                row_index = input('Enter row index: ')
                position = info.loc[info.index==int(row_index), 'Pos'].values[0]
            else:
                position = info['Pos'].values[0]
            s_pos = pos_df.loc[pos_df['position']==position, 'simplePosition'].values[0]
            return s_pos
        except IndexError:
            print(f"Player: {name} not found.")
        return 'UNK'
    def updateMaddenPositions(self):
        df = pd.read_csv("%s.csv" % (self.mr_dir + "allOverallRatings_01-23"))
        # pos = self.getPosition('X dan', 2023, 'BUF')
        # print(pos)
        df = df.loc[df['position']=='UNK_POS']
        df = df.loc[df['abbr']=='GNB']
        df = df.tail(10)
        for index, (name, year, abbr) in enumerate(df[['name', 'year', 'abbr']].values):
            new_pos = self.getPosition(name, year, abbr)
            print(name, new_pos)
            time.sleep(2)
        return
    
# END / FootballDb

#####################

fd = FootballDb("./")

fd.updateMaddenPositions()