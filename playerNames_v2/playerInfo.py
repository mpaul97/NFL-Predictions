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
from retrying import retry
from itertools import cycle

from sportsipy.nfl.roster import Player

pd.options.mode.chained_assignment = None

class PlayerInfo:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.tn_dir = self._dir + "../teamNames/"
        self.pos_dir = self._dir + "../data/positionData/"
        # frames
        self.sp = pd.read_csv("%s.csv" % (self.data_dir + "simplePositions"))
        self.tn = pd.read_csv("%s.csv" % (self.tn_dir + "teamNames"))
        self.player_info: pd.DataFrame = None
        self.rosters: pd.DataFrame = None
        self.position_data: pd.DataFrame = None
        # info
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        self.div_table_containers = [
            'div_passing', 'div_defense', 'div_receiving_and_rushing',
            'div_kicking', 'div_returns', 'div_games_played'
        ]
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
        """
        Build new playerInfo (pid, name, positions(total), years(active)),
        or update when existing
        """
        new_df = pd.DataFrame(columns=['p_id', 'name', 'positions', 'years'])
        if 'playerInfo.csv' in os.listdir(self.data_dir):
            df = pd.read_csv("%s.csv" % (self.data_dir + "playerInfo"))
            new_df = pd.concat([df, new_df])
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
                # updates values if row exists
                idx = len(new_df.index) if pid not in new_df['p_id'].values else new_df.loc[new_df['p_id']==pid].index.values[0]
                new_df.loc[idx] = [pid, name, s_positions, years]
            time.sleep(5)
        new_df.sort_values(by=['p_id'], inplace=True)
        self.saveFrame(new_df, (self.data_dir + "playerInfo"))
        return
    def buildRosters(self):
        new_df = pd.DataFrame(columns=['year', 'abbr', 'pids', 'names'])
        if 'rosters.csv' in os.listdir(self.data_dir):
            df = pd.read_csv("%s.csv" % (self.data_dir + "rosters"))
            new_df = pd.concat([df, new_df])
        abbrs: list[str] = list(self.tn['abbr'].values)
        start_year = 1993 if new_df.empty else 2023
        print('Updating 2023 rosters..') if new_df.empty else print('Building all rosters...')
        for year in range(start_year, 2024):
            for abbr in abbrs:
                try:
                    print(abbr, year)
                    url = "https://www.pro-football-reference.com/teams/" + abbr.lower() + "/" + str(year) + "_roster.htm"
                    res = requests.get(url)
                    start = res.text.index('div_roster')
                    soup = BeautifulSoup(res.text[start:], 'html.parser')
                    table_component = soup.find('table', id="roster")
                    td_tags = table_component.find_all('td', {'data-stat': 'player'})
                    pids, names = [], []
                    for td in td_tags:
                        try:
                            arr = str(td).replace("<b>","").replace("</b>","").split("/")
                            pn = arr[3]
                            pid = pn.split(".htm")[0]
                            name = (pn.split(".htm")[-1]).replace('"','').replace("<",'').replace(">",'')
                            pids.append(pid)
                            names.append(name)
                        except IndexError:
                            pass
                    # idx = len(new_df.index) if start_year == 1993 else new_df.loc[(new_df['year']==year)&(new_df['abbr']==abbr)].index.values[0]
                    new_df.loc[len(new_df.index)] = [year, abbr, '|'.join(pids), '|'.join(names)]
                    time.sleep(5)
                except ValueError:
                    print(f"No table found: {abbr}, {year}")
        self.saveFrame(new_df, (self.data_dir + "rosters"))
        return
    def find_tables_start(self, text: str):
        indices = {}
        for val in self.div_table_containers:
            try:
                indices[val] = text.index(val)
            except ValueError:
                pass
        indices = dict(sorted(indices.items(), key=lambda item: item[1]))
        return list(indices.values())[0] if len(indices) != 0 else 0
    def make_request(self, url: str):
        res = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'})
        res.raise_for_status()
        return res
    def buildPlayerTeams_scrape(self):
        self.setPlayerInfo()
        pdf = self.player_info
        pdf['max_year'] = pdf['years'].apply(lambda x: int(x.split('-')[-1]))
        pdf = pdf.loc[pdf['max_year']>=1993].reset_index(drop=True)
        year_cols = [y for y in range(1993, 2024)]
        all_df = pd.DataFrame(columns=['p_id']+year_cols)
        for index, pid in enumerate(pdf['p_id'].values):
            print(index, len(pdf.index), pid)
            # self.printProgressBar(index, len(df.index), 'Build playerTeams')
            letter = pid[0].upper()
            url = "https://www.pro-football-reference.com/players/" + letter + "/" + pid + ".htm"
            # res = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'})
            res = self.make_request(url)
            start = self.find_tables_start(res.text)
            tables = pd.read_html(res.text[start:])
            df_list = []
            if len(tables) != 0:
                for df in tables:
                    try:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(1)
                        df = df[['Year', 'Tm']]
                        df['Year'].fillna(method='ffill', inplace=True)
                        df_list.append(df)
                    except KeyError:
                        pass
                new_df = pd.concat(df_list)
                new_df.fillna('', inplace=True)
                new_df['Year'] = new_df['Year'].apply(lambda x: str(int(x)) if isinstance(x, float) or isinstance(x, int) else x)
                new_df = new_df.loc[
                    (~new_df['Year'].str.contains('yrs'))&
                    (~new_df['Year'].str.contains('yr'))&
                    (~new_df['Year'].str.contains('Career'))&
                    (~new_df['Tm'].str.contains('TM'))
                ]
                new_df['Year'] = new_df['Year'].apply(lambda x: x.replace('*','').replace('+','')).astype(int)
                new_df.drop_duplicates(inplace=True)
                for year in list(set(new_df['Year'].values)):
                    temp_df: pd.Series = new_df.loc[new_df['Year']==year, 'Tm']
                    if temp_df.shape[0] > 1: # pipe seperate abbrs when played for mulitple teams in same year
                        abbrs = '|'.join(temp_df.values)
                        new_df.drop(temp_df.index.values, inplace=True)
                        new_df.loc[len(new_df.index)] = [year, abbrs]
                new_df.sort_values(by=['Year'], inplace=True)
                all_abbrs = []
                for y in year_cols:
                    try:
                        all_abbrs.append(new_df.loc[new_df['Year']==y, 'Tm'].values[0])
                    except IndexError:
                        all_abbrs.append(np.nan)
                all_df.loc[len(all_df.index)] = [pid] + all_abbrs
                time.sleep(5)
            else:
                print(f"No tables found: {pid}")
        self.saveFrame(all_df, "playerTeams")
        return
    def buildPlayerTeams(self):
        self.setPositionData()
        self.setPlayerInfo()
        self.setRosters()
        df, cd = self.player_info, self.rosters
        df['max_year'] = df['years'].apply(lambda x: int(x.split("-")[-1]))
        df = df.loc[df['max_year']>=1993].reset_index(drop=True)
        year_cols = [y for y in range(1993, 2024)]
        new_df = pd.DataFrame(columns=['p_id']+year_cols)
        for index, pid in enumerate(df['p_id'].values):
            self.printProgressBar(index, len(df.index), 'Building playerTeams')
            info = cd.loc[cd['pids'].str.contains(pid), ['year', 'abbr']]
            all_abbrs = []
            for year in year_cols:
                abbrs = '|'.join(info.loc[info['year']==year, 'abbr'].values)
                all_abbrs.append(abbrs)
            new_df.loc[len(new_df.index)] = [pid] + all_abbrs
        new_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        self.saveFrame(new_df, (self.data_dir + "playerTeams"))
        return
    def updatePlayerTeams(self):
        self.setRosters()
        self.setPlayerInfo()
        self.setPositionData()
        df = pd.read_csv("%s.csv" % (self.data_dir + "playerTeams"))
        self.player_info['max_year'] = self.player_info['years'].apply(lambda x: int(x.split("-")[-1]))
        cd = self.player_info.loc[self.player_info['max_year']>=2023].reset_index(drop=True)
        for index, pid in enumerate(cd['p_id'].values):
            self.printProgressBar(index, len(cd.index), 'Updating playerTeams')
            r_abbrs = self.rosters.loc[(self.rosters['year']==2023)&(self.rosters['pids'].str.contains(pid)), 'abbr'].values
            p_abbrs = list(set(self.position_data.loc[self.position_data['p_id']==pid, 'abbr'].values))
            all_abbrs = '|'.join(list(set(list(r_abbrs)+p_abbrs)))
            try: # player exists in playerTeams
                idx = df.loc[df['p_id']==pid].index.values[0]
                df.at[idx, '2023'] = all_abbrs
            except IndexError:
                df.loc[len(df.index)] = [pid] + [(np.nan if y != 2023 else all_abbrs) for y in range(1993, 2024)]
        df.sort_values(by=['p_id'], inplace=True)
        self.saveFrame(df, (self.data_dir + "playerTeams"))
        return
    def updateAll(self):
        self.buildPlayerInfo()
        self.buildRosters()
        self.updatePlayerTeams()
        return
    def setPositionData(self):
        """
        Returns:
            _type_: ALL 2023 position data
        """
        fns = [fn for fn in os.listdir(self.pos_dir) if '.csv' in fn]
        df_list = []
        for fn in fns:
            df = pd.read_csv(self.pos_dir + fn)
            df_list.append(df.loc[df['wy'].str.contains('2023'), ['p_id', 'wy', 'abbr']])
        self.position_data = pd.concat(df_list)
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
# pi.buildPlayerTeams()

# !!! doesn't contain TRADED players for only 2023
# !!! should work after 2023 concludes
# !!! use positionData to get more info for in-season 2023
# pi.buildRosters()
# pi.buildPlayerTeams()
# pi.updatePlayerTeams()

pi.updateAll()