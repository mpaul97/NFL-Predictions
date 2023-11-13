import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import lxml.html as lh
import regex as re
import time
import math
from ordered_set import OrderedSet

from sportsipy.nfl.roster import Player

pd.options.mode.chained_assignment = None

class PlayerInfo:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.tn_dir = self._dir + "../teamNames/"
        # frames
        self.sp = pd.read_csv("%s.csv" % (self.data_dir + "simplePositions"))
        self.tn = pd.read_csv("%s.csv" % (self.tn_dir + "teamNames"))
        self.player_info: pd.DataFrame = None
        self.rosters: pd.DataFrame = None
        # info
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        return
    def convertPositions(self, positions: str):
        if positions == 'UNK_POS':
            return positions
        arr = positions.split("-")
        s_positions = []
        for p in arr:
            try:
                s_positions.append(self.sp.loc[self.sp['position']==p, 'simplePosition'].values[0])
            except IndexError:
                # print(f"Position not found: {p}")
                continue
        return '-'.join(list(set(s_positions))) if len(s_positions) != 0 else 'UNK_POS'
    def buildPlayerInfo(self):
        new_df = pd.DataFrame(columns=['p_id', 'name', 'positions', 'years'])
        for letter in self.letters:
            url = "https://www.pro-football-reference.com/players/" + letter.upper() + "/"
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            div_component = soup.find('div', id="div_players")
            p_tags = div_component.find_all('p')
            for index, p in enumerate(p_tags):
                self.printProgressBar(index, len(p_tags), letter.upper())
                arr = str(p).replace("<b>","").replace("</b>","").split("/")
                pn = arr[3]
                pid = pn.split(".htm")[0]
                name = (pn.split(".htm")[-1]).replace('"','').replace("<",'').replace(">",'')
                info = arr[4]
                positions = re.findall(r"\(([^\)]+)\)", info)
                positions = positions[0] if len(positions) != 0 else 'UNK_POS'
                s_positions = self.convertPositions(positions)
                years = (info.split(")")[-1]).replace("<","").replace(" ","")
                new_df.loc[len(new_df.index)] = [pid, name, s_positions, years]
            print("\n")
        self.saveFrame(new_df, (self.data_dir + "playerInfo"))
        return
    def buildRosters(self):
        abbrs: list[str] = list(self.tn['abbr'].values)
        new_df = pd.DataFrame(columns=['year', 'abbr', 'players'])
        for year in range(1993, 2024):
            for abbr in abbrs:
                try:
                    print(abbr, year)
                    url = "https://www.pro-football-reference.com/teams/" + abbr.lower() + "/" + str(year) + "_roster.htm"
                    res = requests.get(url)
                    start = res.text.index('div_roster')
                    tables = pd.read_html(res.text[start:])
                    df = tables[0]
                    players = '|'.join([p for p in df['Player'].values if p != 'Team Total'])
                    new_df.loc[len(new_df.index)] = [year, abbr, players]
                    time.sleep(5)
                except ValueError:
                    print(f"No table found: {abbr}, {year}")
        self.saveFrame(new_df, (self.data_dir + "rosters"))
        return
    def buildPlayerTeams(self):
        self.setPlayerInfo()
        self.setRosters()
        df, cd = self.player_info, self.rosters
        years = list(set(cd['year'].values))
        new_df = pd.DataFrame(columns=['p_id']+years)
        for index, (pid, name) in enumerate(df[['p_id', 'name']].values):
            self.printProgressBar(index, len(df.index), 'Player Teams')
            all_teams = []
            for year in years:
                temp_df: pd.DataFrame = cd.loc[cd['year']==year]
                players = '|'.join(temp_df['players'].values)
                year_count = players.count(name)
                if year_count != 0:
                    abbrs = []
                    for abbr, players in temp_df[['abbr', 'players']].values:
                        abbrs.append(abbr) if name in players else ''
                    teams = '|'.join(abbrs)
                else:
                    teams = np.nan
                all_teams.append(teams)
            new_df.loc[len(new_df.index)] = [pid] + all_teams
        self.saveFrame(new_df, (self.data_dir + "playerTeams")) 
        return
    def setPlayerInfo(self):
        self.player_info = pd.read_csv("%s.csv" % (self.data_dir + "playerInfo"))
        return
    def setRosters(self):
        self.rosters = pd.read_csv("%s.csv" % (self.data_dir + "rosters"))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
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
    
# END / PlayerInfo

######################

pi = PlayerInfo("./")

# pi.buildPlayerInfo()
# pi.buildRosters()
pi.buildPlayerTeams()