import pandas as pd
import numpy as np
import os
import time
import urllib.request
import regex as re
from googlesearch import search
from ordered_set import OrderedSet
import math

import sys
sys.path.append('../')
from myRegex.namesRegex import getNames, nameToInfo, teamNameToAbbr, getTeamNames, convertAltAbbr, getKickoffNames, kickoffNameToAbbr

pd.options.mode.chained_assignment = None

class Collect:
    def __init__(self, _dir):
        self._dir = _dir
        self.game_data_dir = self._dir + "../data/"
        self.player_names_dir = self._dir + "../playerNames/"
        self.data_dir = self._dir + "data/"
        self.raw_tables_dir = self.data_dir + "raw/"
        self.clean_tables_dir = self.data_dir + "clean/"
        self.divider_dir = self.data_dir + "dividers/"
        # frames
        self.all_tables: pd.DataFrame = None
        self.player_info: pd.DataFrame = None
        return
    def mostCommon(self, List: list):
        return max(set(List), key = List.index)
    def writeTable(self, url: str, key: str):
        try:
            fp = urllib.request.urlopen(url)
            mybytes = fp.read()
            mystr: str = mybytes.decode("utf8", errors='ignore')
            fp.close()
            start0 = mystr.index('id="div_pbp')
            mystr = mystr[start0:]
            dfs = pd.read_html(mystr, match='Full Play-By-Play Table')
            df = dfs[0]
            # convert/compress table
            for index, row in df.iterrows():
                qr = row['Quarter']
                if type(qr) == str:
                    if 'Quarter' in qr or 'Regulation' in qr:
                        df.drop(index, inplace=True)
            df.to_csv(self.raw_tables_dir + key + ".csv.gz", index=False, compression="gzip")
        except ValueError:
            print("PBP not found at:", url)
        return
    def saveTables(self):
        """
        Saves raw tables for keys not already written but in gameData
        """
        df = pd.read_csv("%s.csv" % (self.game_data_dir + "gameData"))
        keys = [fn.replace('.csv.gz', '') for fn in os.listdir(self.raw_tables_dir)]
        df = df.loc[~df['key'].isin(keys)]
        if len(df.index) != 0:
            print(f"Writing raw tables, length: {len(df.index)}")
        else:
            print("Tables up-to-date.")
        for index, row in df.iterrows():
            key = row['key']
            print(row['wy'], key)
            url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
            self.writeTable(url, key)
            time.sleep(2)
        return
    def cleanRawTables(self):
        """
        Unzip and convert column names of all tables
        """
        cols = ['Quarter', 'Time', 'Down', 'ToGo', 'Location', 'Detail', 'away_points', 'home_points', 'EPB', 'EPA']
        raw_keys = [fn.replace('.csv.gz','') for fn in os.listdir(self.raw_tables_dir)]
        clean_keys = [fn.replace('.csv','') for fn in os.listdir(self.clean_tables_dir)]
        uncleaned_keys = [key for key in raw_keys if key not in clean_keys]
        for index, key in enumerate(uncleaned_keys):
            self.printProgressBar(index, len(uncleaned_keys), 'Cleaning raw tables')
            df = pd.read_csv("%s.csv.gz" % (self.raw_tables_dir + key), compression="gzip")
            abbrs = [(col, index) for index, col in enumerate(df.columns) if len(col) == 3 and col not in ['EPB', 'EPA']]
            abbrs.sort(key=lambda x: x[1])
            n_cols = list(df.columns)
            n_cols[abbrs[0][1]] = 'away_points'
            n_cols[abbrs[1][1]] = 'home_points'
            df.columns = n_cols
            df = df[cols]
            df.columns = [col.lower() for col in df.columns]
            self.saveFrame(df, (self.clean_tables_dir + key))
        return
    def updateAllData(self):
        """
        Add new tables to allTables
        """
        all_df = pd.read_csv("%s.csv" % (self.data_dir + "allTables"))
        clean_keys = [fn.replace('.csv','') for fn in os.listdir(self.clean_tables_dir)]
        unadded_keys = list(set(clean_keys).difference(set(all_df['key'])))
        df_list = []
        # file = open((self.data_dir + "allDetails.txt"), "a")
        for index, key in enumerate(unadded_keys):
            df = pd.read_csv("%s.csv" % (self.clean_tables_dir + key))
            df.insert(0, 'primary_key', [(key + '-' + str(i)) for i in range(len(df.index))])
            df.insert(1, 'key', key)
            df.insert(2, 'num', [i for i in range(len(df.index))])
            df_list.append(df)
            df = df[['detail']]
            df.dropna(inplace=True)
            # file.write('\n'.join(df['detail']))
        # file.close()
        new_df = pd.concat(df_list)
        self.saveFrame(pd.concat([all_df, new_df]), (self.data_dir + "allTables"))
        print(f"allTables updated, length: {len(unadded_keys)}. Rewrite allDetails if needed.")
        return
    def updateTables(self):
        """
        Collects and cleans missing tables keys not in directory
        """
        self.saveTables()
        self.cleanRawTables()
        self.updateAllData()
        return
    def fixTable(self, key: str):
        """
        Used to fix tables that contain columns inline,
        remove bad rows before running
        Args:
            key (str): key to fix
        """
        cd = pd.read_csv("%s.csv" % (self.data_dir + "allTables"))
        cd = cd.loc[cd['key']!=key]
        df = pd.read_csv("%s.csv" % (self.clean_tables_dir + key))
        df.insert(0, 'primary_key', [(key + '-' + str(i)) for i in range(len(df.index))])
        df.insert(1, 'key', key)
        df.insert(2, 'num', [i for i in range(len(df.index))])
        new_df = pd.concat([cd, df])
        new_df.sort_values(by=['key', 'num'], inplace=True)
        self.saveFrame(new_df, (self.data_dir + "allTables"))
        return
    def find_all(self, mystr, sub):
        start = 0
        while True:
            start = mystr.find(sub, start)
            if start == -1: return
            yield start
            start += len(sub) # use start += 1 to find overlapping matches
    def writeDivider(self, url: str, key: str):
        try:
            fp = urllib.request.urlopen(url)
            mybytes = fp.read()
            mystr = mybytes.decode("utf8", errors='ignore')
            fp.close()
            start0 = mystr.index('id="div_pbp')
            mystr = mystr[start0:]
            s_divs = list(self.find_all(mystr, '<tr class="divider" '))
            s1_divs = list(self.find_all(mystr, '<tr class="divider score" '))
            s_divs += s1_divs
            s_divs.sort()
            new_df = pd.DataFrame(columns=['info'])
            for s in s_divs:
                temp1 = mystr[s:]
                end1 = temp1.index('data-stat="exp_pts_before"')
                temp2 = temp1[:end1]
                temp2 = re.sub(r"<[^>]*>", "", temp2)
                names = getNames(temp2, False)
                for name in names:
                    temp2 = temp2.replace(name, '|')
                temp2 = temp2.replace(' (','|')
                temp2 = temp2.replace(' -','|')
                if '|' not in temp2:
                    b_index = temp2.index('<')
                    temp2 = temp2[:b_index] + '|' + temp2[b_index:]
                info = temp2.split('|')[0]
                new_df.loc[len(new_df.index)] = [info]
            new_df.to_csv("%s.csv.gz" % (self.divider_dir + key), index=False, compression='gzip')
        except ValueError:
            print("PBP not found at:", url)
        return
    def saveDividers(self):
        df = pd.read_csv("%s.csv" % (self.game_data_dir + "gameData"))
        keys = [fn.replace('.csv.gz', '') for fn in os.listdir(self.divider_dir)]
        df = df.loc[~df['key'].isin(keys)]
        if len(df.index) != 0:
            print(f"Writing dividers, length: {len(df.index)}")
        else:
            print("Dividers up-to-date.")
        for index, row in df.iterrows():
            key = row['key']
            print(row['wy'], key)
            url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
            self.writeDivider(url, key)
            time.sleep(2)
        return
    def addNames(self):
        """
        Adds names found in details, | seperated
        """
        self.setAllTables()
        df = self.all_tables
        df.fillna('', inplace=True)
        df['names'] = df['detail'].apply(lambda x: '|'.join(getNames(x, False)))
        self.saveFrame(df, (self.data_dir + "allTables"))
        return
    def createAllNames(self):
        self.setAllTables()
        df = self.all_tables
        df.dropna(subset=['names'], inplace=True)
        tn = pd.read_csv("%s.csv" % (self._dir + "../teamNames/teamNames_line"))
        file = open((self.data_dir + "allNames.txt"), "w")
        names = '|'.join(self.all_tables['names'].values).split("|")
        names = list(set(names))
        names = list(set(names).difference(set(tn['name'].values)))
        names.sort()
        file.write('\n'.join(names))
        file.close()
        return
    def getLeadingName(self, row: pd.Series):
        names = row['names'].split("|")
        info = { row['detail'].index(n): n for n in names if n in row['detail'] }
        min_key = min(info.keys())
        return info[min_key]
    def addLeadingNames(self):
        self.setAllTables()
        df = self.all_tables
        df.fillna('', inplace=True)
        df['leading_name'] = df.apply(lambda x: self.getLeadingName(x), axis=1)
        self.saveFrame(df, (self.data_dir + "allTables"))
        return
    def createLeadingPossession(self):
        self.setPlayerInfo()
        pi = self.player_info
        self.setAllTables()
        df = self.all_tables
        df.fillna('', inplace=True)
        cd = pd.read_csv("%s.csv" % (self.game_data_dir + "gameData"))
        df = df.loc[df['key']=='202309100chi']
        for ln in df['leading_name'].values:
            info = pi.loc[(pi['name'].str.contains(ln))|(pi['aka'].str.contains(ln))]
            if info.empty:
                print(ln)
        return
    def getPidsFromName(self, name: str, url: str):
        pattern = r"/players/[A-Z]/.+.htm"
        try: # pfr search works
            fp = urllib.request.urlopen(url)
            mybytes = fp.read()
            mystr = mybytes.decode("utf8", errors='ignore')
            fp.close()
            start = mystr.index('<h1>Search Results</h1>')
            mystr = mystr[start:]
            end = mystr.index('class="search-pagination"')
            mystr = mystr[:end]      
        except ValueError: # pfr search does not work
            all_urls = []
            for i in search(name + ' pro football reference', num=5, stop=5, pause=1):
                if re.search(r"www\.pro-football-reference\.com/players/[A-Z]/", i):
                    all_urls.append(i)
            mystr = '\n'.join(all_urls)
        links = re.findall(pattern, mystr)
        pids = []
        for link in links:
            link = link.split('/')
            pid = link[-1].replace('.htm', '')
            if pid not in pids:
                pids.append(pid)
        return pids
    def getNameFromPid(self, pid: str):
        p_char = pid[0].upper()
        url = 'https://www.pro-football-reference.com/players/' + p_char +'/' + pid + '.htm'
        fp = urllib.request.urlopen(url)
        mybytes = fp.read()
        mystr = mybytes.decode("utf8", errors='ignore')
        fp.close()
        start = mystr.index('id="meta"')
        mystr = mystr[start:]
        start1 = mystr.index('<h1>')
        mystr = mystr[start1:]
        end = mystr.index('</h1>')
        mystr = mystr[:end]
        return [n for n in re.findall(r">([^>]*?)<", mystr.replace('\t','').replace('\n','')) if len(n) != 0][0]
    def getNewYears(self, df: pd.DataFrame):
        curr_years = df['year'].values
        new_years = []
        last_valid_year = 0
        for index, year in enumerate(curr_years):
            if type(year) is float and math.isnan(year):
                if index != 0:
                    new_years.append(last_valid_year)
            else:
                if (type(year) is str or type(year) is object) and '*' in year:
                    year = year.replace('*', '')
                if (type(year) is str or type(year) is object) and '+' in year:
                    year = year.replace('+', '')
                last_valid_year = year
                new_years.append(year)
        return new_years
    def getPlayerInfo(self, pid: str):
        url_c = pid[0].upper()
        url = 'https://www.pro-football-reference.com/players/' + url_c + '/' + pid + '.htm'
        tables = pd.read_html(url)
        df_list = []
        for t in tables:
            if type(t.columns[0]) is tuple or len(tables) == 1 or (type(t.columns[0]) is str and 'Year' in t.columns):
                temp_df = t[t.columns[:4]]
                temp_df.columns = ['year', 'age', 'team', 'position']
                temp_df['year'] = self.getNewYears(temp_df)
                temp_df['year'] = pd.to_numeric(temp_df['year'], errors='coerce')
                temp_df.dropna(subset=['year'], inplace=True)
                df_list.append(temp_df)
        df = pd.concat(df_list)
        df.drop_duplicates(inplace=True)
        df.sort_values(by=['year'], inplace=True)
        df.drop(columns=['age'], inplace=True)
        df.dropna(inplace=True)
        position = self.mostCommon([pos for pos in df['position'].values if 'Missed season' not in pos])
        info = []
        for index, row in df.iterrows():
            year = row['year']
            team = row['team']
            info.append(team + ',' + str(int(year)))
        info = list(OrderedSet(info))
        return position, '|'.join(info)
    def updateFinalPlayerInfo(self):
        self.setPlayerInfo()
        df = self.player_info
        all_names = open((self.data_dir + "allNames.txt"), "r")
        all_names = all_names.read().split("\n")
        missing_names = list(set(all_names).difference(set(df['name'].values)))
        print(f'Updating finalPlayerInfo from allNames.txt, length: {len(missing_names)}')
        for index, name in enumerate(missing_names):
            self.printProgressBar(index, len(missing_names), 'Updating')
            pfr_name = name.lower().replace(' ', '+')
            url = 'https://www.pro-football-reference.com/search/search.fcgi?hint=' + pfr_name + '&search=' + pfr_name
            pids = self.getPidsFromName(name, url)
            time.sleep(3)
            for pid in pids:
                if pid not in df['p_id'].values: # pid doesnt exist insert new row
                    try:
                        pos, info = self.getPlayerInfo(pid)
                        df.loc[len(df.index)] = [pid, name, pos, info, np.nan]
                        time.sleep(3)
                    except ValueError:
                        print(f"No info: {pid}, {name}\n")
                else: # pid in but name UNK, update name
                    idx = df.loc[df['p_id']==pid].index.values[0]
                    pn = df.iloc[idx]['name']
                    if pn == 'UNK':
                        df.at[idx, 'name'] = name
        self.saveFrame(df, (self.player_names_dir + "finalPlayerInfo_1"))
        return
    def addPossessions(self):
        fn = 'possessions'
        df = pd.read_csv("%s.csv" % (self.data_dir + "allTables"))
        t_keys = list(set(df['key']))
        exists = ((fn + '.csv') in os.listdir(self.data_dir))
        pdf = pd.DataFrame()
        if exists:
            pdf = pd.read_csv("%s.csv" % (self.data_dir + fn), low_memory=False)
            keys = list(set(t_keys).difference(set(pdf['key'])))
        else:
            keys = t_keys
        if len(keys) == 0:
            print('Possessions up-to-date.')
            return
        else:
            print(f"Updating possessions, length: {len(keys)}")
        df_list = []
        for index, key in enumerate(keys):
            self.printProgressBar(index, len(keys), 'Updating possessions')
            try:
                df_list.append(self.getPossession(key))
            except UnboundLocalError:
                print(f"Unbound: {key}")
        new_df = pd.concat(df_list)
        if exists:
            new_df = pd.concat([pdf, new_df])
        self.saveFrame(new_df, (self.data_dir + fn))
        return
    def updatePossession(self):
        """
        Collects missing dividers and updates possession
        """
        self.saveDividers()
        self.addPossessions()
        return
    def join(self):
        df = pd.read_csv("%s.csv" % (self.data_dir + "allTables"))
        pdf = pd.read_csv("%s.csv" % (self.data_dir + "possessions"))
        df = df.merge(pdf, on=['primary_key', 'key', 'num'])
        new_df = df.loc[df['key']=='202309100chi', ['num', 'detail', 'possession']]
        new_df.to_csv("temp.csv", index=False)
        return
    def setAllTables(self):
        self.all_tables = pd.read_csv("%s.csv" % (self.data_dir + "allTables"))
        return
    def setPlayerInfo(self):
        self.player_info = pd.read_csv("%s.csv" % (self.player_names_dir + "finalPlayerInfo"))
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
    
# END / Collect

########################

c = Collect("./")

# c.updateTables()

# c.updatePossession()

# c.addNames()
    
# c.createAllNames()

# c.addLeadingNames()

c.updateFinalPlayerInfo()

# c.createLeadingPossession()