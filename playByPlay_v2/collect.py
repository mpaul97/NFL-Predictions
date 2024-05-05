import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import numpy as np
import os
import time
import urllib.request
import regex as re
from googlesearch import search
from ordered_set import OrderedSet
import math
import multiprocessing
from pandas.errors import EmptyDataError
from itertools import repeat

import sys
sys.path.append('../')
from myRegex.spacy_names import getNames
from pbp_names.custom_names import get_names_custom
from pbp_custom_ners.custom_ents import get_custom_ents, get_all_row_ents, ALL_ENTS

pd.options.mode.chained_assignment = None

# MULTIPROCESSING FUNCTIONS
# ---------------------------

def getLeadingName(row: pd.Series):
    """
    Get first name in detail
    Args:
        row (pd.Series): tables row

    Returns:
        _type_: name or NaN
    """
    try:
        names = row['names'].split("|")
        info = { row['detail'].index(n): n for n in names if n in row['detail'] }
        min_key = min(info.keys())
        return info[min_key]
    except (AttributeError, ValueError) as error:
        return np.nan

def func_names(df: pd.DataFrame):
    df['names'] = df['detail'].apply(lambda x: get_names_custom(x) if 'coin toss' not in x else np.nan)
    df['leading_name'] = df.apply(lambda x: getLeadingName(x), axis=1)
    df = df[['primary_key', 'names', 'leading_name']]
    return df

# ---------------------------

class Collect:
    def __init__(self, _dir):
        self._dir = _dir
        self.game_data_dir = self._dir + "../data/"
        self.player_names_dir = self._dir + "../playerNames_v2/data/"
        self.team_names_dir = self._dir + "../teamNames/"
        self.data_dir = self._dir + "data/"
        self.raw_tables_dir = self.data_dir + "raw/"
        self.clean_tables_dir = self.data_dir + "clean/"
        # frames
        self.tn_df: pd.DataFrame = pd.read_csv("%s.csv" % (self.team_names_dir + "teamNames"))
        self.tn_df_pbp: pd.DataFrame = pd.read_csv("%s.csv" % (self.team_names_dir + "teamNames_pbp"))
        self.gd: pd.DataFrame = pd.read_csv("%s.csv" % (self.game_data_dir + "gameData"))
        self.all_tables: pd.DataFrame = None
        self.sample: pd.DataFrame = None
        self.all_table_names: pd.DataFrame = None
        self.sample_names: pd.DataFrame = None
        self.player_info: pd.DataFrame = None
        self.player_teams: pd.DataFrame = None
        self.pcd: pd.DataFrame = None # playerInfo + playerTeams
        self.akas: pd.DataFrame = None
        # info
        self.leading_positions = [
            'QB', 'QB-P', 'QB-WR', 'TE-QB', 'RB-DB', 
            'LB-RB-TE', 'RB-WR', 'TE-RB-LB', 'RB-LB', 'RB', 
            'RB-TE', 'WR-DB', 'RB-WR', 'WR', 'WR-TE', 
            'QB-WR', 'TE-DL-LB', 'TE-LB', 'LB-RB-TE', 'LB-TE', 
            'DL-TE', 'TE-RB-LB', 'TE', 'TE-OL', 'WR-TE', 
            'RB-TE', 'TE-QB', 'OL', 'TE-OL', 'DL-OL',
            'P-K', 'P', 'K'
        ]
        return
    # helpers
    def mostCommon(self, List: list):
        return max(set(List), key = List.index)
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
    def find_all(self, mystr, sub):
        start = 0
        while True:
            start = mystr.find(sub, start)
            if start == -1: return
            yield start
            start += len(sub) # use start += 1 to find overlapping matches
    # END helpers
    # getters/setters
    def getPossessionsDf(self, use_sample: bool = False):
        return pd.read_csv("%s.csv" % (self.data_dir + "sample_possessions")) if use_sample else pd.read_csv("%s.csv" % (self.data_dir + "possessions"))
    def getNamesDf(self, use_sample: bool = False):
        self.setSampleNames() if use_sample else self.setAllTablesNames()
        df = self.sample_names if use_sample else self.all_tables_names
        return df
    def getDf(self, use_sample: bool = False):
        self.setSample() if use_sample else self.setAllTables()
        df = self.sample if use_sample else self.all_tables
        return df
    def getDfWithNames(self, use_sample: bool = False):
        df, ndf = self.getDf(use_sample), self.getNamesDf(use_sample)
        return df.merge(ndf, on=['primary_key'])
    def getDfWithNamesAndPossessions(self, use_sample: bool = False):
        df, ndf = self.getDf(use_sample), self.getNamesDf(use_sample)
        df = df.merge(ndf, on=['primary_key'])
        pdf = self.getPossessionsDf(use_sample)
        if use_sample:
            pdf.drop(columns=['detail'], inplace=True)
        return df.merge(pdf, on=['primary_key'])
    def setPcd(self):
        self.setPlayerInfo(), self.setPlayerTeams(), self.setAkas()
        self.pcd = self.player_info[['p_id', 'positions', 'name']].merge(self.player_teams, on=['p_id'])
        self.pcd = self.pcd.merge(self.akas, on=['p_id'], how='left')
        return
    def setSampleNames(self):
        self.sample_names = pd.read_csv("%s.csv" % (self.data_dir + "sample_names"))
        return
    def setAllTablesNames(self):
        self.all_tables_names = pd.read_csv("%s.csv" % (self.data_dir + "allTables_names"), low_memory=False)
        return
    def setSample(self):
        self.sample = pd.read_csv("%s.csv" % (self.data_dir + "sample"))
        return
    def setAllTables(self):
        self.all_tables = pd.read_csv("%s.csv" % (self.data_dir + "allTables"), low_memory=False)
        return
    def setAkas(self):
        self.akas = pd.read_csv("%s.csv" % (self.player_names_dir + "akas"))
        return
    def setPlayerTeams(self):
        self.player_teams = pd.read_csv("%s.csv" % (self.player_names_dir + "playerTeams"))
        return
    def setPlayerInfo(self):
        self.player_info = pd.read_csv("%s.csv" % (self.player_names_dir + "playerInfo"))
        return
    def createSample(self, key: str):
        self.setAllTables()
        df = self.all_tables.loc[self.all_tables['key']==key]
        self.saveFrame(df, (self.data_dir + "sample"))
        return
    def createAllSampleFrames(self, key: str):
        """
        Create sample from key. Create sample names + leading names.
        Create sample possessions.
        """
        self.createSample(key)
        try:
            self.createDfNames(use_sample=True)
            self.createPossessions(use_sample=True)
        except (EmptyDataError, ValueError) as err:
            pass
        return
    def getAbbrs(self, key: str):
        abbrs = self.gd.loc[self.gd['key']==key, ['home_abbr', 'away_abbr']].values[0]
        return { 'home_abbr': abbrs[0], 'away_abbr': abbrs[1] }
    def getNoNames(self, use_sample: bool = False):
        fn = "sample_noNames" if use_sample else "noNames"
        return pd.read_csv("%s.csv" % (self.data_dir + fn))
    # END getters/setters
    # scraping/writing tables
    def fixAllTables(self, key: str):
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
    def writeTable(self, key: str):
        url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
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
        start = df.loc[df['wy'].str.contains('2011')].index.values[0]
        df: pd.DataFrame = df.loc[df.index>=start].reset_index(drop=True)
        keys = [fn.replace('.csv.gz', '') for fn in os.listdir(self.raw_tables_dir)]
        df = df.loc[~df['key'].isin(keys)]
        if len(df.index) != 0:
            print(f"Writing raw tables, length: {len(df.index)}")
        else:
            print("Tables up-to-date.")
        for index, row in df.iterrows():
            key = row['key']
            print(row['wy'], key)
            self.writeTable(key)
            time.sleep(5)
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
    def createAllTables(self):
        fns = os.listdir(self.clean_tables_dir)
        df_list = []
        for index, fn in enumerate(fns):
            self.printProgressBar(index, len(fns), "Creating allTables")
            df = pd.read_csv(self.clean_tables_dir + fn)
            if df['quarter'].dtype == 'object':
                drop_idxs = df.loc[
                    (~pd.isna(df['quarter']))&
                    ((df['quarter'].str.contains('Quarter'))|(df['quarter'].str.contains('Regulation')))
                ].index.values
                df.drop(drop_idxs, inplace=True)
                df.drop_duplicates(inplace=True)
            key = fn.replace(".csv","")
            df.insert(0, 'primary_key', [(key + '-' + str(i)) for i in range(len(df.index))])
            df.insert(1, 'key', key)
            df.insert(2, 'num', [i for i in range(len(df.index))])
            df_list.append(df)
        new_df = pd.concat(df_list)
        self.saveFrame(new_df, (self.data_dir + "allTables"))
        return
    def updateAllData(self):
        """
        Add new tables to allTables
        """
        all_df = pd.read_csv("%s.csv" % (self.data_dir + "allTables"))
        clean_keys = [fn.replace('.csv','') for fn in os.listdir(self.clean_tables_dir)]
        unadded_keys = list(set(clean_keys).difference(set(all_df['key'])))
        df_list = []
        file = open((self.data_dir + "allDetails.txt"), "a")
        for index, key in enumerate(unadded_keys):
            df = pd.read_csv("%s.csv" % (self.clean_tables_dir + key))
            df.insert(0, 'primary_key', [(key + '-' + str(i)) for i in range(len(df.index))])
            df.insert(1, 'key', key)
            df.insert(2, 'num', [i for i in range(len(df.index))])
            df_list.append(df)
            df = df[['detail']]
            df.dropna(inplace=True)
            file.write('\n'.join(df['detail']))
        file.close()
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
    def findEmptyTableDetails(self):
        df = self.getDf()
        df = df[['key', 'detail']]
        df.fillna('', inplace=True)
        gdf = df.groupby('key').sum()
        gdf = gdf.reset_index()
        gdf['all_details_length'] = gdf['detail'].apply(lambda x: len(x) if isinstance(x, str) else -1)
        gdf.sort_values(by=['all_details_length'], inplace=True)
        print(gdf)
        return
    # END scraping/writing tables
    # names
    def createDfNames(self, use_sample: bool = False):
        start = time.time()
        df = self.getDf(use_sample)
        df = df[['primary_key','detail']]
        df.fillna('', inplace=True)
        num_cores = multiprocessing.cpu_count()-6
        df_split = np.array_split(df, num_cores)
        df_list = []
        if __name__ == '__main__':
            pool = multiprocessing.Pool(num_cores)
            new_df = pd.concat(pool.map(func_names, df_split))
            df_list.append(new_df)
            pool.close()
            pool.join()
        if df_list:
            # Cleanup - remove names in kickoff lines
            new_df = pd.concat(df_list)
            self.saveFrame(new_df, (self.data_dir + ("sample_names" if use_sample else "allTables_names")))
            end = time.time()
            elapsed = end - start
            print("Df Names Time elapsed: {:.2f}".format(elapsed))
        return
    def createAllDetails_txt(self):
        """
        Write all details to .txt file
        """
        df = self.getDf()
        df.dropna(subset=['detail'], inplace=True)
        file = open((self.data_dir + "allDetails.txt"), "w")
        file.write("\n".join(df['detail'].values))
        file.close()
        return
    def createAllNames_txt(self):
        """
        Write all names to .txt file
        """
        df = self.getNamesDf()
        df.dropna(subset=['names'], inplace=True)
        all_names = ('|'.join(df['names'].values)).split("|")
        all_names = list(set(all_names))
        all_names.sort()
        file = open((self.data_dir + "allNames.txt"), "w")
        file.write("\n".join(all_names))
        file.close()
        return
    def checkAllNames(self):
        """
        Find missing names in playerInfo
        """
        self.setPlayerInfo(), self.setAkas()
        file = open((self.data_dir + "allNames.txt"), "r")
        names = file.read().split("\n")
        missing_names = []
        for name in names:
            b_pi = name not in self.player_info['name'].values
            b_tn = name not in '|'.join(self.tn_df_pbp['names'].values)
            b_aka = name.rstrip() not in self.akas['aka'].values # in akas, strip trailing whitespace
            if b_pi and b_tn and b_aka:
                missing_names.append(name)
        if len(missing_names) == 0:
            print("No missing names found in playerInfo, akas, and teamNames_pbp")
            return
        print(f"Found missing names: {missing_names}")
        print(f"Length: {len(missing_names)}")
        return
    def replaceNames(self, row: pd.Series, team_names: list[str]):
        line, names = row[['detail', 'names']]
        try:
            for n in names.split('|'):
                if n not in team_names:
                    line = line.replace(n, '|NAME|')
            return line
        except AttributeError:
            return np.nan
    def createNoNameDetails(self, use_sample: bool = False):
        """
        Create df with '|NAME_0|, |NAME_1|, ...' replacing names; for is off or def player,
        who is the passer, runner, tackler, etc.
        """
        self.setPcd()
        cd = self.getDfWithNames(use_sample)
        keys = self.gd.loc[self.gd['key'].isin(cd['key']), 'key'].values
        team_names = ('|'.join(self.tn_df_pbp['names'].values)).split("|")
        df_list = []
        for index, key in enumerate(keys):
            if not use_sample:
                self.printProgressBar(index, len(keys), "Creating noNames")
            df: pd.DataFrame = cd.loc[cd['key']==key].reset_index(drop=True)
            df['no_name_detail'] = df.apply(lambda x: self.replaceNames(x, team_names), axis=1)
            df = df[['primary_key', 'no_name_detail']]
            df_list.append(df)
        new_df = pd.concat(df_list)
        fn = "sample_noNames" if use_sample else "noNames"
        self.saveFrame(new_df, (self.data_dir + fn))
        return
    def checkNoNames(self):
        df = self.getNoNames()
        df.dropna(inplace=True)
        all_lines = '|'.join(df['no_name_detail'].values)
        names = list(set(re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+", all_lines)))
        for n in names:
            print(n)
        return
    # END names
    # possessions
    def getPossession_coinToss(self, line: str):
        try:
            receiving_name = (re.findall(r"\,\s[A-Z][a-z]+\s", line)[0]).replace(',','').replace(' ','')
        except IndexError:
            receiving_name = (re.findall(r"\,\s49ers", line)[0]).replace(',','').replace(' ','')
        return self.tn_df.loc[self.tn_df['name'].str.contains(receiving_name), 'abbr'].values[0]
    def getPossession_play(self, line: str, leading_name: str, year: str, abbrs: dict):
        # leading_name IN teamNames_pbp -> timeout, challenge, etc. (no possession change) -> nan
        if leading_name in '|'.join(self.tn_df_pbp['names'].values):
            return np.nan
        # leading_name in playerInfo
        df = self.pcd
        players = df.loc[
            (df['name'].str.contains(leading_name)|(df['aka'].str.contains(leading_name)))&
            (~pd.isna(df[year]))&
            ((df[year].str.contains(abbrs['home_abbr']))|(df[year].str.contains(abbrs['away_abbr'])))&
            (df['positions'].isin(self.leading_positions)),
            ['p_id', year]
        ]
        if players.shape[0] == 1:
            return [a for a in abbrs.values() if a in players.values[0, 1]][0]
        return np.nan
    def getDetailPossession(self, row: pd.Series, abbrs: dict):
        """
        Get possession from leading name or num==0(kickoff) else NaN
        Args:
            row (pd.Series): tables row
            abbrs (list[str]): home + away abbr
        Returns:
            _type_: home_abbr, away_abbr, or NaN
        """
        num, key, line, leading_name = row[['num', 'key', 'detail', 'leading_name']]
        if num == 0: # coin toss
            return self.getPossession_coinToss(line)
        if pd.isna(leading_name):
            return np.nan
        if 'Penalty' in line and line.index('Penalty') == 0: # No play, just penalty
            return np.nan
        year = (self.gd.loc[self.gd['key']==key, 'wy'].values[0]).split(" | ")[1]
        return self.getPossession_play(line, leading_name, year, abbrs)
    def func_possessions(self, df: pd.DataFrame):
        df_list = []
        for key in list(set(df['key'])):
            temp_df: pd.DataFrame = df.loc[df['key']==key].reset_index(drop=True)
            abbrs = { 'home_abbr': temp_df.iloc[0]['home_abbr'], 'away_abbr': temp_df.iloc[0]['away_abbr'] }
            temp_df['possession'] = temp_df.apply(lambda x: self.getDetailPossession(x, abbrs), axis=1)
            temp_df['possession'].fillna(method='ffill', inplace=True)
            df_list.append(temp_df)
        return pd.concat(df_list)
    def createPossessions(self, use_sample: bool = False):
        """
        Gets current team in possession of ball, using leading names,
        ffill -> fill NaN values with last valid entry
        Args:
            use_sample (bool, optional): use sample DataFrame. Defaults to False.
        """
        start = time.time()
        self.setPcd()
        df = self.getDfWithNames(use_sample)
        df = df[['primary_key', 'num', 'key', 'detail', 'leading_name']]
        num_cores = multiprocessing.cpu_count()-6
        df = df.merge(self.gd[['key', 'home_abbr', 'away_abbr']], on=['key'])
        df_split = np.array_split(df, num_cores)
        df_list = []
        if __name__ == '__main__':
            pool = multiprocessing.Pool(num_cores)
            new_df = pd.concat(pool.map(self.func_possessions, df_split))
            df_list.append(new_df)
            pool.close()
            pool.join()
        if df_list:
            cols = ['primary_key', 'detail', 'possession'] if use_sample else ['primary_key', 'possession']
            fn = "sample_possessions" if use_sample else "possessions"
            all_df = df[['primary_key']].merge(pd.concat(df_list)[cols], on=['primary_key'], how='left')
            all_df['possession'].fillna(method='ffill', inplace=True)
            self.saveFrame(all_df, (self.data_dir + fn))
            end = time.time()
            elapsed = end - start
            print("Possessions Time elapsed: {:.2f}".format(elapsed))
        return
    # END possessions
    # entities
    def testEntities(self, ent_name: str, use_sample: bool = False):
        df = self.getDf(use_sample)
        df = df[['primary_key', 'detail']]
        new_df = pd.DataFrame(columns=list(df.columns)+ALL_ENTS)
        for index, (primary_key, detail) in enumerate(df[['primary_key', 'detail']].values):
            new_df.loc[len(new_df.index)] = [primary_key, detail] + get_all_row_ents(detail)
        lines = new_df.loc[~(pd.isna(new_df[ent_name])), 'detail'].values
        for line in lines:
            print(line)
        return
    def createEntities(self, use_sample: bool = False):
        df = self.getDf(use_sample)
        df = df[['primary_key', 'detail']]
        new_df = pd.DataFrame(columns=list(df.columns)+ALL_ENTS)
        for index, (primary_key, detail) in enumerate(df[['primary_key', 'detail']].values):
            self.printProgressBar(index, len(df.index), 'Creating entities')
            new_df.loc[len(new_df.index)] = [primary_key, detail] + get_all_row_ents(detail)
        fn = "sample_entities" if use_sample else "allEntities"
        self.saveFrame(new_df, (self.data_dir + fn))
        return
    # END entities
    def fixInvalidTables(self):
        """
        Find tables with no/short details, rewrite, clean,
        remove from allTables, sort
        """
        self.findEmptyTableDetails()
        # self.writeTable('202211030htx')
        # self.cleanRawTables()
        # self.updateAllData()
        # df = self.getDf()
        # df.sort_values(by=['key', 'num'], inplace=True)
        # self.saveFrame(df, (self.data_dir + "allTables"))
        return

# END / Collect

########################

# c = Collect("./")

# c.createAllTables()

# c.saveTables()
# c.cleanRawTables()
# c.updateTables()
