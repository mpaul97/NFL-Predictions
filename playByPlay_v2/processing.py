import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import numpy as np
import os
import time
import regex
import re
from ordered_set import OrderedSet
import math
import multiprocessing
from pandas.errors import EmptyDataError
from itertools import repeat
import requests
from bs4 import BeautifulSoup
import datetime
import random

import sys
sys.path.append('../')
from pbp_names.custom_names import get_names_custom
from pbp_custom_ners.custom_ents import get_custom_ents, get_all_row_ents, ALL_ENTS
# from play_info.penalties import Penalties

pd.options.mode.chained_assignment = None

def getLeadingName(row: pd.Series):
    """
    Get first name in detail
    Args:
        row (pd.Series): tables row
    Returns:
        str/float: name or NaN
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

class Processing:
    def __init__(self, _dir: str, testing: bool = False):
        self.testing = testing
        # paths
        self._dir = _dir
        self.game_data_dir = self._dir + "../data/"
        self.player_names_dir = self._dir + "../playerNames_v2/data/"
        self.team_names_dir = self._dir + "../teamNames/"
        self.data_dir = self._dir + "data/"
        # frames
        self.tn_df: pd.DataFrame = pd.read_csv("%s.csv" % (self.team_names_dir + "teamNames"))
        self.tn_df_pbp: pd.DataFrame = pd.read_csv("%s.csv" % (self.team_names_dir + "teamNames_pbp"))
        self.gd: pd.DataFrame = pd.read_csv("%s.csv" % (self.game_data_dir + "gameData"))
        self.player_data: pd.DataFrame = None
        # other
        self.leading_positions = [
            'QB', 'QB-P', 'QB-WR', 'TE-QB', 'RB-DB', 
            'LB-RB-TE', 'RB-WR', 'TE-RB-LB', 'RB-LB', 'RB', 
            'RB-TE', 'WR-DB', 'RB-WR', 'WR', 'WR-TE', 
            'QB-WR', 'TE-DL-LB', 'TE-LB', 'LB-RB-TE', 'LB-TE', 
            'DL-TE', 'TE-RB-LB', 'TE', 'TE-OL', 'WR-TE', 
            'RB-TE', 'TE-QB', 'OL', 'TE-OL', 'DL-OL',
            'P-K', 'P', 'K'
        ]
        self.PID_ENTS = ALL_ENTS.copy()
        self.PID_ENTS.remove('TEAM_NAME')
        # ENTS that are DEFINTELY offensive players
        self.OFF_ENTS = ['PASSER', 'RECEIVER', 'RUSHER']
        # ENTS that are DEFINTELY defensive players
        self.DEF_ENTS = ['DEFENDER', 'INTERCEPTOR', 'SACKER']
        # offensive pid_ents
        self.offensive_pid_ents = [
            'pid_PASSER', 'pid_RECEIVER', 'pid_RUSHER', 
            'pid_KICKER', 'pid_PUNTER', 'pid_RETURNER'
        ]
        # offensive penalties
        self.offensive_penalties = [
            'False Start', 'Illegal Block', 'Ineligible Downfield'
        ]
        return
    # helpers
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def print_progress_bar(self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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
    def get_datetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def add_datetime_columns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.get_datetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def flatten(self, l: list):
        return [x for xs in l for x in xs]
    # END helpers
    # getters/setters
    def get_player_info(self):
        """
        Get DataFrame playerInfo
        """
        return pd.read_csv("%s.csv" % (self.player_names_dir + "playerInfo"))
    def get_player_teams(self):
        """
        Get DataFrame playerTeams
        """
        return pd.read_csv("%s.csv" % (self.player_names_dir + "playerTeams"))
    def get_akas(self):
        """
        Get DataFrame akas
        """
        return pd.read_csv("%s.csv" % (self.player_names_dir + "akas"))
    def set_player_data(self):
        """
        Set player_data as merge of playerInfo, playerTeams, and akas
        """
        self.player_data = self.get_player_info()[['p_id', 'positions', 'name']].merge(self.get_player_teams(), on=['p_id'])
        self.player_data = self.player_data.merge(self.get_akas(), on=['p_id'], how='left')
        return 
    def get_pbp(self):
        """
        Get pbp table, sample or all
        Returns:
            pd.DataFrame: pbp table(s)
        """
        try:
            return pd.read_csv("%s.csv" % (self.data_dir + "sample")) if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables"))
        except FileNotFoundError:
            print("pbp does not exist.")
    def get_names_pbp(self):
        """
        Get pbp names table, sample or all
        Returns:
            pd.DataFrame: pbp names table(s)
        """
        try:
            return pd.read_csv("%s.csv" % (self.data_dir + "sample_names")) if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables_names"))
        except FileNotFoundError:
            print('names pbp does not exist.')
    def get_pbp_and_names_pbp(self):
        """
        Get pbp + names pbp table, sample or all
        Returns:
            pd.DataFrame: pbp + names pbp table(s)
        """
        return self.get_pbp().merge(self.get_names_pbp(), on=['primary_key'])
    def get_possessions_pbp(self):
        """
        Get pbp possessions table, sample or all
        Returns:
            pd.DataFrame: pbp possessions table(s)
        """
        try:
            return pd.read_csv("%s.csv" % (self.data_dir + "sample_possessions"))[['primary_key', 'possession']] if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables_possessions"))
        except FileNotFoundError:
            print('possessions pbp does not exist.')
    def get_entities_pbp(self):
        """
        Get pbp entities table, sample or all
        Returns:
            pd.DataFrame: pbp entities table(s)
        """
        try:
            return pd.read_csv("%s.csv" % (self.data_dir + "sample_entities")) if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables_entities"), low_memory=False)
        except FileNotFoundError:
            print("entities pbp does not exist.")
    def get_pids_pbp(self):
        """
        Get pbp pids table, sample or all
        Returns:
            pd.DataFrame: pbp pids table(s)
        """
        try:
            return pd.read_csv("%s.csv" % (self.data_dir + "sample_pids")) if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables_pids"), low_memory=False)
        except FileNotFoundError:
            print("pids pbp does not exist.")
    def get_pid_entities_pbp(self):
        """
        Get pbp pid_entities table, sample or all
        Returns:
            pd.DataFrame: pbp pid_entities table(s)
        """
        try:
            return pd.read_csv("%s.csv" % (self.data_dir + "sample_pid_entities")) if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables_pid_entities"))
        except FileNotFoundError:
            print("pid_entities pbp does not exist.")
    def get_pids_pbp_and_pid_entities_pbp(self):
        """
        Get total (pids + pid_entities) pbp table, sample or all
        Returns:
            pd.DataFrame: pbp total table(s)
        """
        pbp, pe = self.get_pids_pbp(), self.get_pid_entities_pbp()
        return pbp.merge(pe, on=['primary_key'])
    def get_pids_pbp_and_pid_entities_pbp_and_possessions_pbp(self):
        """
        Get total (pids + pid_entities + possessions) pbp table, sample or all
        Returns:
            pd.DataFrame: pbp total table(s)
        """
        pbp, pe, pos = self.get_pids_pbp(), self.get_pid_entities_pbp(), self.get_possessions_pbp()
        pbp = pbp.merge(pe, on=['primary_key'])
        return pbp.merge(pos, on=['primary_key'])
    def get_pid_entities_pbp_and_possessions_pbp(self):
        """
        Get pbp pid_entities table, sample or all + possessions
        Returns:
            pd.DataFrame: pbp pid_entities table(s)
        """
        try:
            df: pd.DataFrame = pd.read_csv("%s.csv" % (self.data_dir + "sample_pid_entities")) if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables_pid_entities"))
            return df.merge(self.get_possessions_pbp(), on='primary_key')
        except FileNotFoundError:
            print("pid_entities pbp does not exist.")
    def get_pbp_and_possessions_pbp(self):
        """
        Get total (pbp + possessions) pbp table, sample or all
        Returns:
            pd.DataFrame: pbp total table(s)
        """
        pbp, pdf = self.get_pbp(), self.get_possessions_pbp()
        return pbp.merge(pdf, on=['primary_key'])
    def get_pbp_and_pids_pbp(self):
        """
        Get total (pbp + pids) pbp table, sample or all
        Returns:
            pd.DataFrame: pbp total table(s)
        """
        pbp, pids = self.get_pbp(), self.get_pids_pbp()
        return pbp.merge(pids, on=['primary_key'])
    def get_pbp_and_pid_entities_pbp(self):
        """
        Get total (pbp + pid_entities) pbp table, sample or all
        Returns:
            pd.DataFrame: pbp total table(s)
        """
        pbp, pdf = self.get_pbp(), self.get_pid_entities_pbp()
        return pbp.merge(pdf, on=['primary_key'])
    def get_pbp_and_entities_and_pids_pbp(self):
        """
        Get total (pbp + entities + pids) pbp table, sample or all
        Returns:
            pd.DataFrame: pbp total table(s)
        """
        pbp, pbp_entities, pids = self.get_pbp(), self.get_entities_pbp(), self.get_pids_pbp()
        pbp = pbp.merge(pbp_entities, on=['primary_key'])
        return pbp.merge(pids, on=['primary_key'])
    def get_total_pbp(self):
        """
        Get total (pbp + names_pbp + possessions_pbp + entities + pids) pbp table, sample or all
        Returns:
            pd.DataFrame: pbp total table(s)
        """
        pbp, pbp_names, pbp_poss, pbp_entities = self.get_pbp(), self.get_names_pbp(), self.get_possessions_pbp(), self.get_entities_pbp()
        pids, p_ents = self.get_pids_pbp(), self.get_pid_entities_pbp()
        pbp = pbp.merge(pbp_names, on=['primary_key'])
        pbp = pbp.merge(pbp_poss, on=['primary_key'])
        pbp = pbp.merge(pbp_entities, on=['primary_key'])
        pbp = pbp.merge(pids, on=['primary_key'])
        return pbp.merge(p_ents, on=['primary_key'])
    def get_game_pids_abbrs(self):
        """
        Get game pids abbrs, sample or all
        Returns:
            pd.DataFrame: game pids abbrs table
        """
        try:
            return pd.read_csv("%s.csv" % (self.data_dir + "sample_game_pids_abbrs")) if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables_game_pids_abbrs"))
        except FileNotFoundError:
            print("game_pids_abbrs does not exist.")
    def get_pbp_missing_pid_abbrs(self):
        """
        Get pbp_missing_pid_abbrs, sample or all
        Returns:
            pd.DataFrame: pbp_missing_pid_abbrs table
        """
        try:
            return pd.read_csv("%s.csv" % (self.player_names_dir + "pbp_missing_pid_abbrs"))
        except FileNotFoundError:
            print("pbp_missing_pid_abbrs does not exist.")
    def get_pid_entities_team_abbrs(self):
        """
        Get pid entities team abbrs, sample or all
        Returns:
            pd.DataFrame: pid entities team abbrs table
        """
        try:
            return pd.read_csv("%s.csv" % (self.data_dir + "sample_pid_entities_team_abbrs")) if self.testing else pd.read_csv("%s.csv" % (self.data_dir + "allTables_pid_entities_team_abbrs"))
        except FileNotFoundError:
            print("_pid_entities_team_abbrs does not exist.")
    # END getters/setters
    # names
    def build_allDetails_txt(self):
        """
        Write all details to .txt file
        """
        df = self.get_pbp()
        df.dropna(subset=['detail'], inplace=True)
        file = open((self.data_dir + "allDetails.txt"), "w")
        file.write("\n".join(df['detail'].values))
        file.close()
        return
    def build_names(self):
        """
        Write PBP names to CSV. Columns = [primary_key, names, leading_name]
        """
        start = time.time()
        df = self.get_pbp()
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
            new_df = pd.concat(df_list)
            self.save_frame(new_df, (self.data_dir + ("sample_names" if self.testing else "allTables_names")))
            end = time.time()
            elapsed = end - start
            print("PBP Names Time elapsed: {:.2f}".format(elapsed))
        # return
    def build_allNames_txt(self):
        """
        Write all names to .txt file
        """
        df = self.get_names_pbp()
        df.dropna(subset=['names'], inplace=True)
        all_names = ('|'.join(df['names'].values)).split("|")
        all_names = list(set(all_names))
        all_names.sort()
        file = open((self.data_dir + "allNames.txt"), "w")
        file.write("\n".join(all_names))
        file.close()
        return
    def check_names(self):
        """
        Find missing names in playerInfo
        """
        pi, akas = self.get_player_info(), self.get_akas()
        file = open((self.data_dir + "allNames.txt"), "r")
        names = file.read().split("\n")
        missing_names = []
        for name in names:
            b_pi = name not in pi['name'].values
            b_tn = name not in '|'.join(self.tn_df_pbp['names'].values)
            b_aka = name.rstrip() not in akas['aka'].values # in akas, strip trailing whitespace
            if b_pi and b_tn and b_aka:
                missing_names.append(name)
        if len(missing_names) == 0:
            print("No missing names found in playerInfo, akas, and teamNames_pbp")
            return
        print(f"Found missing names: {missing_names}")
        print(f"Length: {len(missing_names)}")
        return
    def update_names(self):
        """
        Update allTables_names and allNames.txt for new keys
        """
        df = self.get_pbp()
        df_copy = df.copy()
        ndf = self.get_names_pbp()
        # df with primary_keys in pbp but not in names_pbp
        df = df.loc[~df['primary_key'].isin(ndf['primary_key'])]
        if df.empty:
            print("allTables_names up-to-date.")
            return
        new_keys = list(set(df['key']))
        print(f"Updating allTables_names for keys: {new_keys}")
        df = df[['primary_key','detail']]
        df.fillna('', inplace=True)
        df = func_names(df)
        ndf = pd.concat([ndf, df])
        ndf = df_copy[['primary_key']].merge(ndf, on=['primary_key'])
        self.save_frame(ndf, (self.data_dir + "allTables_names"))
        return
    # END names
    # possession
    def get_possession_coin_toss(self, line: str):
        """
        Get possession for coin toss, receiving team name
        Args:
            line (str): detail

        Returns:
            str: abbr
        """
        try:
            receiving_name = (regex.findall(r"\,\s[A-Z][a-z]+\s", line)[0]).replace(',','').replace(' ','')
        except IndexError:
            receiving_name = (regex.findall(r"\,\s49ers", line)[0]).replace(',','').replace(' ','')
        return self.tn_df.loc[self.tn_df['name'].str.contains(receiving_name), 'abbr'].values[0]
    def get_possession_play(self, leading_name: str, year: str, abbrs: dict):
        """
        Get possession for normal plays by cross referencing player_data to
        find player name = leading_name, and abbr in year
        Args:
            leading_name (str): first name in detail
            year (str): wy year
            abbrs (dict): home + away abbrs
        Returns:
            _type_: abbr or NaN
        """
        # leading_name IN teamNames_pbp -> timeout, challenge, etc. (no possession change) -> nan
        if leading_name in '|'.join(self.tn_df_pbp['names'].values):
            return np.nan
        # leading_name in playerInfo
        df = self.player_data
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
    def get_possession(self, row: pd.Series, abbrs: dict):
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
            return self.get_possession_coin_toss(line)
        if pd.isna(leading_name):
            return np.nan
        if 'Penalty' in line and line.index('Penalty') == 0: # No play, just penalty
            return np.nan
        year = (self.gd.loc[self.gd['key']==key, 'wy'].values[0]).split(" | ")[1]
        return self.get_possession_play(leading_name, year, abbrs)
    def func_possessions(self, df: pd.DataFrame):
        df_list = []
        for key in list(set(df['key'])):
            temp_df: pd.DataFrame = df.loc[df['key']==key].reset_index(drop=True)
            abbrs = { 'home_abbr': temp_df.iloc[0]['home_abbr'], 'away_abbr': temp_df.iloc[0]['away_abbr'] }
            temp_df['possession'] = temp_df.apply(lambda x: self.get_possession(x, abbrs), axis=1)
            temp_df['possession'].fillna(method='ffill', inplace=True)
            df_list.append(temp_df)
        return pd.concat(df_list)
    def build_possessions(self):
        """
        Write PBP possessions to CSV. Columns = [primary_key, possession]
        Gets current team in possession of ball, using leading names,
        ffill -> fill NaN values with last valid entry
        """
        start = time.time()
        self.set_player_data()
        df = self.get_pbp_and_names_pbp()
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
            cols = ['primary_key', 'detail', 'possession'] if self.testing else ['primary_key', 'possession']
            fn = "sample_possessions" if self.testing else "allTables_possessions"
            all_df = df[['primary_key']].merge(pd.concat(df_list)[cols], on=['primary_key'], how='left')
            all_df['possession'].fillna(method='ffill', inplace=True)
            self.save_frame(all_df, (self.data_dir + fn))
            end = time.time()
            elapsed = end - start
            print("Possessions Time elapsed: {:.2f}".format(elapsed))
        return
    def update_possessions(self):
        """
        Update allTables_possessions
        """
        self.set_player_data()
        df = self.get_pbp_and_names_pbp()
        df_copy = df.copy()
        pdf = self.get_possessions_pbp()
        # df with primary_keys in pbp but not in possessions_pbp
        df = df.loc[~df['primary_key'].isin(pdf['primary_key'])]
        if df.empty:
            print("allTables_possessions up-to-date.")
            return
        new_keys = list(set(df['key']))
        print(f"Updating allTables_possessions for keys: {new_keys}")
        df = df[['primary_key', 'num', 'key', 'detail', 'leading_name']]
        df = df.merge(self.gd[['key', 'home_abbr', 'away_abbr']], on=['key'])
        df = self.func_possessions(df)
        df = df[['primary_key', 'possession']]
        pdf: pd.DataFrame = pd.concat([pdf, df])
        pdf['possession'].fillna(method='ffill', inplace=True)
        pdf = df_copy[['primary_key']].merge(pdf, on=['primary_key'])
        self.save_frame(pdf, (self.data_dir + "allTables_possessions"))
        return
    # END possession
    # entities
    def test_entities(self, ent_name: str):
        df = self.get_pbp()
        df = df[['primary_key', 'detail']]
        new_df = pd.DataFrame(columns=list(df.columns)+ALL_ENTS)
        for index, (primary_key, detail) in enumerate(df[['primary_key', 'detail']].values):
            new_df.loc[len(new_df.index)] = [primary_key, detail] + get_all_row_ents(detail)
        lines = new_df.loc[~(pd.isna(new_df[ent_name])), 'detail'].values
        for line in lines:
            print(line)
        return
    def func_entities(self, df: pd.DataFrame):
        df[ALL_ENTS] = df.apply(lambda row: get_all_row_ents(row['detail']), axis=1, result_type="expand")
        return df
    def build_entities(self):
        """
        Write PBP entities to CSV. Columns = [primary_key, ALL_ENTS]
        """
        start = time.time()
        df = self.get_pbp()
        df['detail'].fillna('', inplace=True)
        df = df[['primary_key', 'detail']]
        num_cores = multiprocessing.cpu_count()-6
        df_split = np.array_split(df, num_cores)
        df_list = []
        if __name__ == '__main__':
            pool = multiprocessing.Pool(num_cores)
            new_df = pd.concat(pool.map(self.func_entities, df_split))
            df_list.append(new_df)
            pool.close()
            pool.join()
        if df_list:
            new_df = pd.concat(df_list)
            new_df.drop(columns=['detail'], inplace=True)
            new_df = df[['primary_key']].merge(new_df, on=['primary_key'], how='left')
            fn = "sample_entities" if self.testing else "allTables_entities"
            self.save_frame(new_df, (self.data_dir + fn))
            end = time.time()
            elapsed = end - start
            print("Entities Time elapsed: {:.2f}".format(elapsed))
        return
    def update_entities(self):
        """
        Update allTables_entities
        """
        df = self.get_pbp()
        df_copy = df.copy()
        edf = self.get_entities_pbp()
        # df with primary_keys in pbp but not in entities_pbp
        df = df.loc[~df['primary_key'].isin(edf['primary_key'])]
        if df.empty:
            print("allTables_entities up-to-date.")
            return
        new_keys = list(set(df['key']))
        print(f"Updating allTables_entities for keys: {new_keys}")
        df = self.func_entities(df)
        df = df[['primary_key']+ALL_ENTS]
        edf: pd.DataFrame = pd.concat([edf, df])
        edf = df_copy[['primary_key']].merge(edf, on=['primary_key'])
        self.save_frame(edf, (self.data_dir + "allTables_entities"))
        return
    def check_entities(self):
        """
        Check if entity names are correct/in player_data
        """
        # convert entities to names
        df = self.get_entities_pbp()
        vals = (df[ALL_ENTS].values).flatten()
        vals = vals[~pd.isna(vals)]
        splitter = lambda x: x.split(":")[0]
        vfunc = np.vectorize(splitter)
        names = list(set(vfunc(vals)))
        # check if name in player_data
        pi, akas = self.get_player_info(), self.get_akas()
        missing_names = []
        for name in names:
            b_pi = name not in pi['name'].values
            b_tn = name not in '|'.join(self.tn_df_pbp['names'].values)
            b_aka = name.rstrip() not in akas['aka'].values # in akas, strip trailing whitespace
            if b_pi and b_tn and b_aka:
                missing_names.append(name)
        if len(missing_names) == 0:
            print("No missing names found in playerInfo, akas, and teamNames_pbp")
            return
        print(f"Found missing names: {missing_names}")
        print(f"Length: {len(missing_names)}")
        return
    def get_wrong_enitity_lines(self):
        """
        Get array of lines where incorrect entity names occur
        """
        # convert entities to names
        df = self.get_entities_pbp()
        vals = (df[ALL_ENTS].values).flatten()
        vals = vals[~pd.isna(vals)]
        splitter = lambda x: x.split(":")[0]
        vfunc = np.vectorize(splitter)
        names = list(set(vfunc(vals)))
        # check if name in player_data
        df.fillna('', inplace=True)
        df['all_ents'] = df.apply(lambda row: '|'.join(row[ALL_ENTS]), axis=1)
        # file = open((self.data_dir + "allDetails.txt"), "r")
        # all_lines = (file.read()).split("\n")
        cd = self.get_pbp()
        pi, akas = self.get_player_info(), self.get_akas()
        wrong_lines = []
        for name in names:
            b_pi = name not in pi['name'].values
            b_tn = name not in '|'.join(self.tn_df_pbp['names'].values)
            b_aka = name.rstrip() not in akas['aka'].values # in akas, strip trailing whitespace
            if b_pi and b_tn and b_aka:
                # line = [l for l in all_lines if name in l][0]
                # print(f"\"{line}\",")
                try:
                    prim_keys = df.loc[df['all_ents'].str.contains(f"{name}:"), 'primary_key'].values
                    lines = cd.loc[df['primary_key'].isin(prim_keys), 'detail'].values
                    for line in lines[:10]:
                        print(f"\"{line}\",")
                except re.error:
                    continue
        return
    # END entities
    # pids
    def get_pid_details(self, key: str):
        url = "https://www.pro-football-reference.com/boxscores/" + key + ".htm"
        text = requests.get(url).text
        start = text.index('div_pbp')
        soup = BeautifulSoup(text[start:], 'html.parser')
        df = pd.DataFrame(columns=['key', 'detail', 'pids_detail'])
        for index, tag in enumerate(soup.find_all('td', {'data-stat': 'detail'})):
            line = str(tag)
            detail = re.sub(r"[\<].*?[\>]", "", line)
            links = tag.findChildren('a')
            for link in links:
                link = str(link)
                if "href" in link: # replace pid links with raw pids
                    p, t = '/players/', '/teams/'
                    if p in link:
                        start = link.index(p)
                        end = link.index('.htm')
                        pid = link[start+len(p)+2:end]
                        line = line.replace(link, pid)
                    if t in link:
                        abbr = re.findall(r"[A-Z]{3}", line)[0]
                        line = line.replace(link, abbr)
                else: # remove links without href tags
                    line = line.replace(link, "")
            # remove style tags - <i>, <b>
            line = re.sub(r"<\/{0,1}i>", "", line)
            line = re.sub(r"<\/{0,1}b>", "", line)
            # content of td tag (detail)
            line = line[line.index(">")+1:]
            line = line[:line.index("<")]
            df.loc[len(df.index)] = [key, detail, line]
        return df
    def build_pids(self):
        """
        Get pids from url
        """
        df = self.get_pbp()
        keys = list(set(df['key']))
        keys.sort()
        df_list = []
        for index, key in enumerate(keys):
            self.print_progress_bar(index, len(keys), 'Building pids')
            try:
                df_list.append(self.get_pid_details(key))
            except Exception as err:
                print(f"Table error: {err} for key: {key}")
            time.sleep(5)
        new_df = pd.concat(df_list)
        new_df = df[['primary_key', 'key', 'detail']].merge(new_df, on=['key', 'detail'], how='left')
        new_df.drop_duplicates(inplace=True)
        new_df = new_df[['primary_key', 'pids_detail']]
        fn = "sample_pids" if self.testing else "allTables_pids"
        self.save_frame(new_df, (self.data_dir + fn))
        return
    def update_pids(self):
        """
        Update allTables_pids
        """
        df = self.get_pbp()
        df_copy = df.copy()
        pdf = self.get_pids_pbp()
        # df with primary_keys in pbp but not in pids_pbp
        df = df.loc[~df['primary_key'].isin(pdf['primary_key'])]
        if df.empty:
            print("allTables_pids up-to-date.")
            return
        keys = list(set(df['key']))
        print(f"Updating pids for keys: {keys}")
        keys.sort()
        df_list = []
        for key in keys:
            try:
                df_list.append(self.get_pid_details(key))
            except Exception as err:
                print(f"Table error: {err} for key: {key}")
            time.sleep(5)
        new_df = pd.concat(df_list)
        new_df = df[['primary_key', 'key', 'detail']].merge(new_df, on=['key', 'detail'], how='left')
        new_df.drop_duplicates(inplace=True)
        new_df = new_df[['primary_key', 'pids_detail']]
        pdf = pd.concat([pdf, new_df])
        pdf = df_copy[['primary_key']].merge(pdf, on=['primary_key'])
        print(f"Shape: {pdf.shape}")
        confirmation = input("Confirm update? (y/n)")
        if confirmation == 'y':
            self.save_frame(pdf, (self.data_dir + "allTables_pids"))
        return
    def shift_pids_detail(self, df: pd.DataFrame, index: int):
        after_details = df.loc[df.index>=index, 'pids_detail'].values
        before_details = df.loc[df.index<index, 'pids_detail'].values
        new_details = list(before_details) + [np.nan] + list(after_details)[:-1]
        df['pids_detail'] = new_details
        return df
    def fix_pids(self):
        """
        Realign pids so they match pbp details (OT games)
        """
        df, pids = self.get_pbp(), self.get_pids_pbp()
        ot_keys = list(set(df.loc[df['quarter']=='OT', 'key'].values))
        if "fixed_pids_keys.csv" in os.listdir(self.data_dir):
            fdf = pd.read_csv("%s.csv" % (self.data_dir + "fixed_pids_keys"))
            ot_keys = [k for k in ot_keys if k not in fdf['key'].values]
            if len(ot_keys) == 0:
                print("All pids ot games already fixed.")
                return
        else: # nothing fixed yet
            fdf = pd.DataFrame(columns=['key'])
        print(f"Fixing pids for keys: {ot_keys}")
        df_list = []
        for key in ot_keys:
            print(f"Realigning pids for key: {key}")
            temp_df: pd.DataFrame = df.loc[df['key']==key]
            temp_pids: pd.DataFrame = pids.loc[pids['primary_key'].str.contains(key)]
            temp_df.reset_index(drop=True, inplace=True)
            tdf = temp_df.merge(temp_pids, on=['primary_key'])
            overtime_start_index = tdf.loc[tdf['quarter']=='Overtime'].index.values[0]
            self.shift_pids_detail(tdf, overtime_start_index)
            overtime_end_index = tdf.loc[tdf['quarter']=='End of Overtime'].index.values[0]
            self.shift_pids_detail(tdf, overtime_end_index)
            df_list.append(tdf)
            fdf.loc[len(fdf.index)] = [key]
        new_pids = pd.concat(df_list)[['primary_key', 'pids_detail']]
        pids = df[['primary_key', 'key']].merge(pids, on=['primary_key'])
        org_pids = pids.loc[~pids['key'].isin(ot_keys)]
        org_pids.drop(columns=['key'], inplace=True)
        new_df = pd.concat([org_pids, new_pids])
        df = df[['primary_key']].merge(new_df[['primary_key', 'pids_detail']], on=['primary_key'])
        fn = "sample_pids" if self.testing else "allTables_pids"
        print(f"Shape: {df.shape}")
        confirmation = input("Confirm pids fix? (y/n)")
        if confirmation == 'y':
            self.save_frame(df, (self.data_dir + fn))
            self.save_frame(fdf, (self.data_dir, "fixed_pids_keys"))
        return
    # END pids
    # pid entities
    def entities_to_pid(self, row: pd.Series):
        cols = [f"pid_{c}" for c in self.PID_ENTS]
        try:
            detail, pids_detail = row['detail'], row['pids_detail']
            if not pd.isna(detail) and not pd.isna(pids_detail):
                ents = row[self.PID_ENTS].to_dict()
                ents = { key: ents[key] for key in ents if not pd.isna(ents[key]) }
                ent_objs = []
                for key in ents:
                    val = ents[key]
                    for item in val.split("|"):
                        name = item.split(":")[0]
                        start, end = int(item.split(":")[1]), int(item.split(":")[2])
                        ent_objs.append({ "ent_type": key, "name": name, "start": start, "end": end })
                ent_objs.sort(key=lambda x: x['start'])
                s0 = detail
                removed_count = 0
                pids_arr = pids_detail.split(" ")
                for ent in ent_objs:
                    start, end = ent['start'], ent['end']
                    substring: str = detail[start:end]
                    num_spaces = substring.count(" ")
                    sr, er = start-removed_count, end-removed_count
                    ent_key = (substring.replace(" ","") + str(start))
                    s0 = s0[:sr] + ent_key + s0[er:]
                    ent['index'] = [i for i, val in enumerate(s0.split(" ")) if ent_key in val][0]
                    ent['pid'] = pids_arr[ent['index']].replace("(", "").replace(")", "").replace(",", "")
                    ent['pid'] = ent['pid'].rstrip('.').rstrip(':')
                    removed_count += num_spaces - len(str(start))
                _dict = {}
                for col in cols:
                    pids = [f"{obj['pid']}:{obj['start']}:{obj['end']}" for obj in ent_objs if obj['ent_type'] == col.replace("pid_", "")]
                    _dict[col] = '|'.join(pids) if len(pids) != 0 else np.nan
                return _dict
            else:
                return { col: np.nan for col in cols }
        except Exception as err:
            print(row['primary_key'])
            return { col: np.nan for col in cols }
    def build_pid_entities(self):
        """
        Convert entity names to their corresponding pid
        """
        start = time.time()
        df = self.get_pbp_and_entities_and_pids_pbp()
        cols = [f"pid_{c}" for c in self.PID_ENTS]
        df[cols] = df.apply(lambda x: self.entities_to_pid(x), axis=1, result_type="expand")
        df = df[['primary_key']+cols]
        fn = "sample_pid_entities" if self.testing else "allTables_pid_entities"
        self.save_frame(df, (self.data_dir + fn))
        end = time.time()
        elapsed = end - start
        print("Pid Entities Time elapsed: {:.2f}".format(elapsed))
        return
    def update_pid_entities(self):
        """
        Update allTables_pid_entities
        """
        df = self.get_pbp_and_entities_and_pids_pbp()
        df_copy = df.copy()
        edf = self.get_pid_entities_pbp()
        # df with primary_keys in pbp but not in pid_entities_pbp
        df = df.loc[~df['primary_key'].isin(edf['primary_key'])]
        if df.empty:
            print("allTables_pid_entities up-to-date.")
            return
        new_keys = list(set(df['key']))
        print(f"Updating allTables_pid_entities for keys: {new_keys}")
        df = df.loc[df['key'].isin(new_keys)]
        cols = [f"pid_{c}" for c in self.PID_ENTS]
        df[cols] = df.apply(lambda x: self.entities_to_pid(x), axis=1, result_type="expand")
        df = df[['primary_key']+cols]
        edf: pd.DataFrame = pd.concat([edf, df])
        edf = df_copy[['primary_key']].merge(edf, on=['primary_key'])
        print(f"Shape: {edf.shape}")
        confirmation = input("Confirm update? (y/n)")
        if confirmation == 'y':
            self.save_frame(edf, (self.data_dir + "allTables_pid_entities"))
        return
    # END pids entities
    # game pids abbrs
    def get_unknown_game_pids(self, df: pd.DataFrame, pid: str, key: str, home_abbr: str, away_abbr: str):
        """
        Gets pid abbr for players not in playerNames, using pid_ents
        Returns:
            str: abbr
        """
        temp_df: pd.DataFrame = df.loc[df['primary_key'].str.contains(key)]
        mask = temp_df.astype(str).apply(lambda x: x.str.contains(pid))
        contain_cols = mask.any()
        contain_cols = contain_cols[contain_cols].index.tolist()
        contain_cols.remove('pids_detail')
        if len(contain_cols) > 1 and 'pid_PENALIZER' in contain_cols:
            contain_cols.remove('pid_PENALIZER')
        temp_df.fillna('', inplace=True)
        info = temp_df.loc[temp_df[contain_cols[0]].str.contains(pid), ['pids_detail', 'possession']].values[0]
        off_abbr = info[-1]
        def_abbr = list(set([home_abbr, away_abbr]).difference([info[-1]]))[0]
        if contain_cols[0] == 'pid_PENALIZER':
            try:
                pen: str = re.findall(pid + r":\s[A-Z][a-z]+\s[A-Z][a-z]+", info[0])[0]
                pen = pen.replace((pid + ': '), '')
                abbr = off_abbr if 'Offensive' in pen or pen in self.offensive_penalties else def_abbr
                # print(pen)
                # print(f"Off: {off_abbr}, Def: {def_abbr}")
                # print(abbr)
            except IndexError:
                abbr = 'UNK'
        else:
            abbr = off_abbr if contain_cols[0] in self.offensive_pid_ents else def_abbr
        return abbr
    def build_game_pids_abbrs(self):
        """
        Stores all pids per game with corresponding team abbr from pid_entities
        Stores all unknown players
        """
        self.set_player_data()
        df = self.get_pids_pbp_and_pid_entities_pbp_and_possessions_pbp()
        df['key'] = df['primary_key'].apply(lambda x: x.split("-")[0])
        info = self.gd.loc[self.gd['key'].isin(df['key']), ['key', 'wy', 'home_abbr', 'away_abbr']]
        cols = [f"pid_{c}" for c in self.PID_ENTS]
        new_df = pd.DataFrame(columns=['key', 'year', 'pids'])
        missing_df = pd.DataFrame(columns=['key', 'wy', 'p_id', 'abbr'])
        for index, (key, wy, home_abbr, away_abbr) in enumerate(info.values):
            self.print_progress_bar(index, info.shape[0], 'Building game pids abbrs...')
            vals: np.ndarray = (df.loc[df['key']==key, cols].values).flatten()
            vals = vals[~pd.isna(vals)]
            vals = self.flatten([v.split("|") for v in vals])
            pids: list[str] = list(set([v.split(":")[0] for v in vals]))
            year = wy.split(" | ")[1]
            data = []
            for pid in pids:
                try:
                    pid_abbrs = self.player_data.loc[self.player_data['p_id']==pid, year].values[0]
                    if not pd.isna(pid_abbrs):
                        pid_abbrs = pid_abbrs.split("|")
                        if len(pid_abbrs) == 1:
                            data.append(f"{pid}:{pid_abbrs[0]}")
                        else:
                            valid_abbrs = [a for a in pid_abbrs if a in [home_abbr, away_abbr]]
                            if len(valid_abbrs) == 1:
                                data.append(f"{pid}:{valid_abbrs[0]}")
                            else:
                                abbr = self.get_unknown_game_pids(df, pid, key, home_abbr, away_abbr)
                                missing_df.loc[len(missing_df.index)] = [key, wy, pid, abbr]
                                data.append(f"{pid}:{abbr}")
                    else:
                        abbr = self.get_unknown_game_pids(df, pid, key, home_abbr, away_abbr)
                        missing_df.loc[len(missing_df.index)] = [key, wy, pid, abbr]
                        data.append(f"{pid}:{abbr}")
                except IndexError:
                    abbr = self.get_unknown_game_pids(df, pid, key, home_abbr, away_abbr)
                    missing_df.loc[len(missing_df.index)] = [key, wy, pid, abbr]
                    data.append(f"{pid}:{abbr}")
            new_df.loc[len(new_df.index)] = [key, year, '|'.join(data)]
        fn = 'sample_game_pids_abbrs' if self.testing else 'allTables_game_pids_abbrs'
        self.save_frame(new_df, (self.data_dir + fn))
        print("Update UNK manually!")
        if not self.testing:
            self.save_frame(missing_df, (self.player_names_dir + "pbp_missing_pid_abbrs"))
        return
    def update_game_pids_abbrs(self):
        """
        Update game pids abbrs and pbp_missing_pid_abbrs
        """
        self.set_player_data()
        df = self.get_pids_pbp_and_pid_entities_pbp_and_possessions_pbp()
        df['key'] = df['primary_key'].apply(lambda x: x.split("-")[0])
        info = self.gd.loc[self.gd['key'].isin(df['key']), ['key', 'wy', 'home_abbr', 'away_abbr']]
        cd = self.get_game_pids_abbrs()
        info: pd.DataFrame = info.loc[~info['key'].isin(cd['key'])]
        if info.empty:
            print("game_pids_abbrs up-to-date.")
            return
        print(f"Updating game_pids_abbrs and pbp_missing_pid_abbrs for keys: {info['key'].values}")
        cols = [f"pid_{c}" for c in self.PID_ENTS]
        missing_df = self.get_pbp_missing_pid_abbrs()
        for index, (key, wy, home_abbr, away_abbr) in enumerate(info.values):
            vals: np.ndarray = (df.loc[df['key']==key, cols].values).flatten()
            vals = vals[~pd.isna(vals)]
            vals = self.flatten([v.split("|") for v in vals])
            pids: list[str] = list(set([v.split(":")[0] for v in vals]))
            year = wy.split(" | ")[1]
            data = []
            for pid in pids:
                try:
                    pid_abbrs = self.player_data.loc[self.player_data['p_id']==pid, year].values[0]
                    if not pd.isna(pid_abbrs):
                        pid_abbrs = pid_abbrs.split("|")
                        if len(pid_abbrs) == 1:
                            data.append(f"{pid}:{pid_abbrs[0]}")
                        else:
                            valid_abbrs = [a for a in pid_abbrs if a in [home_abbr, away_abbr]]
                            if len(valid_abbrs) == 1:
                                data.append(f"{pid}:{valid_abbrs[0]}")
                            else:
                                abbr = self.get_unknown_game_pids(df, pid, key, home_abbr, away_abbr)
                                missing_df.loc[len(missing_df.index)] = [key, wy, pid, abbr]
                                data.append(f"{pid}:{abbr}")
                    else:
                        abbr = self.get_unknown_game_pids(df, pid, key, home_abbr, away_abbr)
                        missing_df.loc[len(missing_df.index)] = [key, wy, pid, abbr]
                        data.append(f"{pid}:{abbr}")
                except IndexError:
                    abbr = self.get_unknown_game_pids(df, pid, key, home_abbr, away_abbr)
                    missing_df.loc[len(missing_df.index)] = [key, wy, pid, abbr]
                    data.append(f"{pid}:{abbr}")
            cd.loc[len(cd.index)] = [key, year, '|'.join(data)]
        self.save_frame(cd, (self.data_dir + 'allTables_game_pids_abbrs'))
        if not self.testing:
            self.save_frame(missing_df, (self.player_names_dir + "pbp_missing_pid_abbrs"))
        return
    # END game pids abbrs
    # pids entities + abbrs
    def pid_entities_team_abbr(self, row: pd.Series, df: pd.DataFrame):
        key: str = str(row['primary_key']).split("-")[0]
        pids: list[str] = str(df.loc[df['key']==key, 'pids'].values[0]).split("|")
        pe: str = row['pid_entity']
        return str([p for p in pids if pe.split(":")[0] in p][0]).split(":")[1]
    def build_pid_entities_team_abbrs(self):
        """
        Store team_abbr for each pids entities, per game
        Columns: primary_key -> pid:start:end
        """
        N = 15 # max length of pids under same entity type
        pdf = self.get_pid_entities_pbp()
        pdf_copy = pdf.copy()
        df = self.get_game_pids_abbrs()
        # melt all pid_entities
        cols = [f"pid_{c}" for c in self.PID_ENTS]
        pdf = pdf.melt(id_vars=['primary_key'], value_vars=cols, var_name='variable', value_name='pid_entity')[['primary_key', 'pid_entity']]
        pdf.dropna(subset=['pid_entity'], inplace=True)
        # melt all pipe (|) delimited values
        new_cols = [f"col_{i}" for i in range(N)]
        pdf[new_cols] = pdf.apply(lambda x: x['pid_entity'].split("|") + [np.nan for _ in range(N-len(x['pid_entity'].split("|")))], axis=1, result_type="expand")
        pdf: pd.DataFrame = pdf[['primary_key']+new_cols]
        pdf = pdf.melt(id_vars=['primary_key'], value_vars=new_cols, var_name='variable', value_name='pid_entity')[['primary_key', 'pid_entity']]
        pdf.dropna(subset=['pid_entity'], inplace=True)
        # realign
        pdf = pdf_copy[['primary_key']].merge(pdf, on=['primary_key'])
        pdf['abbr'] = pdf.apply(lambda row: self.pid_entities_team_abbr(row, df), axis=1)
        fn = 'sample_pid_entities_team_abbrs' if self.testing else 'allTables_pid_entities_team_abbrs'
        self.save_frame(pdf, (self.data_dir + fn))
        return
    def update_pid_entities_team_abbrs(self):
        """
        Store team_abbr for each pids entities, per game
        Columns: primary_key -> pid:start:end
        """
        cd = self.get_pid_entities_team_abbrs()
        N = 15 # max length of pids under same entity type
        pdf = self.get_pid_entities_pbp()
        pdf_copy = pdf.copy()
        df = self.get_game_pids_abbrs()
        # melt all pid_entities
        cols = [f"pid_{c}" for c in self.PID_ENTS]
        pdf = pdf.melt(id_vars=['primary_key'], value_vars=cols, var_name='variable', value_name='pid_entity')[['primary_key', 'pid_entity']]
        pdf.dropna(subset=['pid_entity'], inplace=True)
        # melt all pipe (|) delimited values
        new_cols = [f"col_{i}" for i in range(N)]
        pdf[new_cols] = pdf.apply(lambda x: x['pid_entity'].split("|") + [np.nan for _ in range(N-len(x['pid_entity'].split("|")))], axis=1, result_type="expand")
        pdf: pd.DataFrame = pdf[['primary_key']+new_cols]
        pdf = pdf.melt(id_vars=['primary_key'], value_vars=new_cols, var_name='variable', value_name='pid_entity')[['primary_key', 'pid_entity']]
        pdf.dropna(subset=['pid_entity'], inplace=True)
        # check if new primary_keys in pdf
        pdf = pdf.loc[~pdf['primary_key'].isin(cd['primary_key'])]
        if pdf.empty:
            print("pid_entities_team_abbrs up-to-date.")
            return
        pks = list(set([pk.split("-")[0] for pk in pdf['primary_key'].values]))
        print(f"Updating pid_entities_team_abbrs for primary_keys: {pks}")
        # realign
        pdf = pdf_copy[['primary_key']].merge(pdf, on=['primary_key'])
        pdf['abbr'] = pdf.apply(lambda row: self.pid_entities_team_abbr(row, df), axis=1)
        pdf = pd.concat([cd, pdf])
        fn = 'sample_pid_entities_team_abbrs' if self.testing else 'allTables_pid_entities_team_abbrs'
        self.save_frame(pdf, (self.data_dir + fn))
        return
    # END pids entities + abbrs
    def update(self):
        # testing must be False
        self.testing = False
        self.update_names()
        self.build_allNames_txt()
        self.build_allDetails_txt()
        self.update_possessions()
        self.update_entities()
        self.update_pids()
        self.fix_pids()
        self.update_pid_entities()
        self.update_game_pids_abbrs()
        self.update_pid_entities_team_abbrs()
        return
    
# END / Processing

#######################

if __name__ == "__main__":
    p = Processing(
        _dir="./",
        testing=False
    )
    p.build_entities()
    p.build_pid_entities()
    p.build_game_pids_abbrs()
    p.build_pid_entities_team_abbrs()
