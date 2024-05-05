import pandas as pd
import numpy as np
import os
import random
import regex as re
import itertools
import time

from penalties import Penalties, PenaltyObject

pd.options.mode.chained_assignment = None

class Conversion:
    def __init__(self, df: pd.DataFrame, _dir: str):
        self.df = df
        self.pt_bool_cols: list[str] = [
            'is_fumble', 'is_penalty', 'is_challenge', 
            'is_block', 'contains_lateral'
        ]
        self._dir = _dir
        self.all_tables: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../data/allTables"))
        self.pos_df: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../data/allTables_possessions"))
        self.pe_df: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../data/allTables_pid_entities"))
        self.pid_ent_cols = [col for col in self.pe_df.columns if re.search(r"pid\_[A-Z]+", col)]
        self.pe_abbrs_df: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../data/allTables_pid_entities_team_abbrs"))
        # frame for getting offensive or defensive touchdown
        self.info_df: pd.DataFrame = self.pe_abbrs_df.merge(self.pe_df, on=['primary_key'])
        self.info_df: pd.DataFrame = self.info_df.merge(self.pos_df, on=['primary_key'])
        self.gd: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../../data/gameData"))
        self.tn: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../../teamNames/teamNames_firstName"))
        self.tn_pbp: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../../teamNames/teamNames_pbp"))
        self.all_cols = {
            'pass': ['completed', 'pass_yards', 'is_interception', 'is_spike'],
            'run': ['rush_yards'],
            'sack': ['sack_yards'],
            'kickoff': ['kickoff_yards', 'return_yards', 'is_touchback'],
            'extra_point': ['is_good'],
            'coin_toss': ['kicking_abbr', 'receiving_abbr', 'winning_abbr'],
            'field_goal': ['is_good', 'field_goal_yards'],
            'punt': ['punt_yards', 'return_yards', 'is_fair_catch', 'is_muffed', 'is_touchback'],
            'timeout': ['timeout_abbr', 'timeout_number'],
            'kneel': [],
            'penalty': []
        }
        self.general_cols = ['is_off_touchdown', 'is_def_touchdown', 'penalty_yards']
        self.cols = self.general_cols + list(set(self.flatten(list(self.all_cols.values()))))
        self.normal_play_type_funcs = {
            'pass': self.normal_pass, 'run': self.normal_run, 'sack': self.normal_sack,
            'kickoff': self.normal_kickoff, 'extra_point': self.normal_extra_point, 'coin_toss': self.normal_coin_toss,
            'field_goal': self.normal_field_goal, 'punt': self.normal_punt, 'penalty': self.normal_penalty,
            'kneel': self.normal_kneel, 'timeout': self.normal_timeout
        }
        # self.funcs = {
        #     'normal': self.normal, 'is_penalty': self.penalty
        # }
        self.funcs = {
            'normal': self.normal
        }
        return
    def flatten(self, l: list):
        return [x for xs in l for x in xs]
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def get_scorer_pid(self, ents: np.ndarray, line: str):
        try:
            td = re.search(r"touchdown", line)
            td_start: int = td.start()
            ents: list[tuple] = [(e.split(":")[0], int(e.split(":")[1]), int(e.split(":")[2])) for e in ents]
            min_ent: tuple = ents[0]
            for pid, start, end in ents:
                if (td_start - end) < (td_start - min_ent[2]):
                    min_ent = (pid, start, end)
            return min_ent[0]
        except Exception:
            return 'UNK'
    # normal funcs
    def normal_pass(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with pass_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
        vals: list[str] = re.findall(r"for\s[-]?[0-9]+", line)
        if len(vals) >= 1:
            _dict['pass_yards'] = int(vals[0].replace("for ",""))
        if 'incomplete' in line or 'spiked' in line or 'no gain' in line:
            _dict['pass_yards'] = 0
        _dict['completed'] = ((len(vals) != 0) and ('intercepted' not in line))
        _dict['is_off_touchdown'] = (('touchdown' in line) and ('returned' not in line))
        _dict['is_def_touchdown'] = (('touchdown' in line) and ('returned' in line))
        _dict['is_interception'] = ('intercepted' in line)
        _dict['is_spike'] = ('spiked' in line)
        return _dict
    def normal_run(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with run_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
        vals: list[str] = re.findall(r"for\s[-]?[0-9]+", line)
        if len(vals) >= 1:
            _dict['rush_yards'] = int(vals[0].replace("for ",""))
        _dict['is_off_touchdown'] = (('touchdown' in line) and ('returned' not in line))
        _dict['is_def_touchdown'] = (('touchdown' in line) and ('returned' in line))
        return _dict
    def normal_sack(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with sack_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
        vals: list[str] = re.findall(r"for\s[-]?[0-9]+", line)
        if len(vals) >= 1:
            _dict['sack_yards'] = int(vals[0].replace("for ",""))
        _dict['is_def_touchdown'] = (('touchdown' in line) and ('returned' in line))
        return _dict
    def normal_kickoff(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with kickoff_cols and general_cols
        Kicking team => offense / Receiving team => defense
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
        kick_vals: list[str] = re.findall(r"kicks\soff\s[-]?[0-9]+", line)
        if len(kick_vals) >= 1:
            _dict['kickoff_yards'] = int(kick_vals[0].replace("kicks off ",""))
        return_vals: list[str] = re.findall(r"for\s[-]?[0-9]+", line)
        if len(return_vals) >= 1:
            _dict['return_yards'] = int(return_vals[0].replace("for ",""))
        _dict['is_touchback'] = ('touchback' in line)
        if 'touchdown' in line:
            ents = row[self.pid_ent_cols].values
            ents = ents[~pd.isna(ents)]
            scorer: str = self.get_scorer_pid(ents, row['pids_detail'])
            if scorer != 'UNK': # if scorer pid -> find if on offense or defense
                scorer: str = scorer.split(" ")[0]
                abbrs: np.ndarray = self.info_df.loc[
                    (self.info_df['primary_key']==row['primary_key'])&
                    (self.info_df['pid_entity'].str.contains(scorer)),
                    ['abbr', 'possession']
                ].values[0]
                _dict['is_off_touchdown'] = (abbrs[0] == abbrs[1])
                _dict['is_def_touchdown'] = (abbrs[0] != abbrs[1])
        return _dict
    def normal_extra_point(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with extra_point_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
        _dict['is_good'] = ('extra point good' in line)
        return _dict
    def normal_coin_toss(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with coin_toss_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
        key: str = (row['primary_key']).split("-")[0]
        leading_name: str = line.split(" ")[0]
        winning_abbr: str = self.tn.loc[self.tn['name']==leading_name, 'abbr'].values[0]
        _dict['winning_abbr'] = winning_abbr
        losing_abbr: str = list(set(self.gd.loc[self.gd['key']==key, ['home_abbr', 'away_abbr']].values[0]).difference(set([winning_abbr])))[0]
        _dict['kicking_abbr'] = winning_abbr if 'deferred' in line else losing_abbr
        _dict['receiving_abbr'] = losing_abbr if 'deferred' in line else winning_abbr
        return _dict
    def normal_field_goal(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with field_goal_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
        _dict['is_good'] = ('field goal good' in line)
        vals: list[str] = re.findall(r"[-]?[0-9]+\syard", line)
        if len(vals) >= 1:
            _dict['field_goal_yards'] = int(vals[0].replace(" yard",""))
        if 'touchdown' in line:
            ents = row[self.pid_ent_cols].values
            ents = ents[~pd.isna(ents)]
            scorer: str = self.get_scorer_pid(ents, row['pids_detail'])
            if scorer != 'UNK': # if scorer pid -> find if on offense or defense
                scorer: str = scorer.split(" ")[0]
                abbrs: np.ndarray = self.info_df.loc[
                    (self.info_df['primary_key']==row['primary_key'])&
                    (self.info_df['pid_entity'].str.contains(scorer)),
                    ['abbr', 'possession']
                ].values[0]
                _dict['is_off_touchdown'] = (abbrs[0] == abbrs[1])
                _dict['is_def_touchdown'] = (abbrs[0] != abbrs[1])
        return _dict
    def normal_punt(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with punt_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
        # ['punt_yards', 'return_yards', 'is_fair_catch', 'is_muffed', 'is_touchback']
        punt_vals: list[str] = re.findall(r"punts\s[-]?[0-9]+", line)
        if len(punt_vals) >= 1:
            _dict['punt_yards'] = int(punt_vals[0].replace("punts ",""))
        return_vals: list[str] = re.findall(r"for\s[-]?[0-9]+", line)
        if len(return_vals) >= 1:
            _dict['return_yards'] = int(return_vals[0].replace("for ",""))
        _dict['is_fair_catch'] = ('fair catch' in line)
        _dict['is_muffed'] = ('muffed' in line)
        _dict['is_touchback'] = ('touchback' in line)
        if 'touchdown' in line:
            ents = row[self.pid_ent_cols].values
            ents = ents[~pd.isna(ents)]
            if len(ents) == 1: # "JoneDo21 punts no gain, touchdown" -> no scorer
                _dict['is_off_touchdown'] = False
                _dict['is_def_touchdown'] = True
            else: # punt team is offense
                scorer: str = self.get_scorer_pid(ents, row['pids_detail'])
                if scorer != 'UNK': # if scorer pid -> find if on offense or defense
                    scorer: str = scorer.split(" ")[0]
                    abbrs: np.ndarray = self.info_df.loc[
                        (self.info_df['primary_key']==row['primary_key'])&
                        (self.info_df['pid_entity'].str.contains(scorer)),
                        ['abbr', 'possession']
                    ].values[0]
                    _dict['is_off_touchdown'] = (abbrs[0] == abbrs[1])
                    _dict['is_def_touchdown'] = (abbrs[0] != abbrs[1])
        return _dict
    def normal_penalty(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with penalty_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        line = row['pids_detail']
        print(line)
        # !!!! PLACEHOLDER !!!!
        # never exists -> penalty means is_penalty -> True
        return _dict
    def normal_kneel(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with kneel_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        # NO attributes
        return _dict
    def normal_timeout(self, _dict: dict, row: pd.Series):
        """
        Returns fille _dict with timeout_cols and general_cols
        Args:
            _dict (dict): all fields
            line (str): pids_detail
        Returns:
            dict: _dict
        """
        # 'timeout_abbr', 'timeout_number'
        line: str = row['pids_detail']
        team_name: str = line.split(" by ")[1]
        abbr: str = self.tn_pbp.loc[self.tn_pbp['names'].str.contains(team_name), 'abbr'].values[0]
        _dict['timeout_abbr'] = abbr
        num: int = int((re.findall(r"\#[0-9]", line)[0]).replace('#',''))
        print(line)
        print(abbr, num)
        return _dict
    # end normal funcs
    def normal_lambda_func(self, row: pd.Series, has_penalty: bool = False):
        """
        Normal pass line to normal attributes conversion
        Args:
            row (pd.Series): DF row
        Returns:
            dict: self.cols
        """
        play_type = row['play_type']
        line: str = row['pids_detail']
        if has_penalty:
            line = line[:line.index('Penalty')]
        _dict = { col: np.nan for col in self.cols }
        if not pd.isna(play_type):
            try:
                return self.normal_play_type_funcs[play_type](_dict, row)
            except KeyError:
                return _dict
        return _dict
    def normal(self, df: pd.DataFrame):
        """
        Convert normal lines
        Attributes: self.cols
        Args:
            df (pd.DataFrame): pass (all bools -> False)
        """
        df[self.cols] = df.apply(lambda x: self.normal_lambda_func(x), axis=1, result_type='expand')
        return df
    # penalties
    def penalty_c1(self, pens: list[PenaltyObject], row: pd.Series):
        """
        Get penalty attributes for case 1: EXACTLY 1 PENALTY
        Args:
            pens (list[PenaltyObject]): penalties
            row (pd.Series): row
        """
        _dict = { col: np.nan for col in self.cols }
        if not pens[0].declined: # one accepted penalty
            _dict['is_touchdown'] = False
            _dict['is_interception'] = False
            _dict['is_spike'] = False
            _dict['penalty_yards'] = pens[0].yards * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        # declined
        return self.normal_lambda_func(row, has_penalty=True)
    def penalty_c2(self, pens: list[PenaltyObject], row: pd.Series):
        """
        Get penalty attributes for case 2: EXACTLY 2 PENALTIES
        Args:
            pens (list[PenaltyObject]): penalties
            row (pd.Series): row
        """
        _dict = { col: np.nan for col in self.cols }
        if all([p.declined for p in pens]): # both declined
            return self.normal_lambda_func(row, has_penalty=True)
        elif pens[0].offset and pens[1].offset: # both offsetting
            if any([p.no_play for p in pens]): # no play
                _dict['is_touchdown'] = False
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                return _dict
            # offsetting penalties but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict 
        elif all([not p.declined for p in pens]): # both accepted (always same team)
            _dict['is_touchdown'] = False
            _dict['is_interception'] = False
            _dict['is_spike'] = False
            _dict['penalty_yards'] = (pens[0].yards + pens[1].yards) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        # one declined, one accepted
        pens = [p for p in pens if not p.declined]
        _dict['is_touchdown'] = False
        _dict['is_interception'] = False
        _dict['is_spike'] = False
        _dict['penalty_yards'] = pens[0].yards * (-1 if pens[0].against_possessing_team else 1)
        return _dict
    def penalty_c3(self, pens: list[PenaltyObject], row: pd.Series):
        """
        Get penalty attributes for case 3: EXACTLY 3 PENALTIES
        Args:
            pens (list[PenaltyObject]): penalties
            row (pd.Series): row
        """
        _dict = { col: np.nan for col in self.cols }
        line: str = row['pids_detail']
        if all([p.declined for p in pens]): # all declined
            return self.normal_lambda_func(row, has_penalty=True)
        elif all([p.offset for p in pens]): # all offsetting
            if any([p.no_play for p in pens]): # no play
                _dict['is_touchdown'] = False
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                return _dict
            # offsetting penalties but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict 
        elif any([p.no_play for p in pens]): # no play
            _dict['is_touchdown'] = False
            _dict['is_interception'] = False
            _dict['is_spike'] = False
            return _dict
        elif all([not p.declined for p in pens]): # all accepted (ASSUMING same team) + play stands
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        elif len([p for p in pens if p.declined]) == 2 and len([p for p in pens if not p.declined]) == 1: # two declined + one accepted
            accepted_pen = [p for p in pens if not p.declined][0]
            if any([p.no_play for p in pens]):
                _dict['is_touchdown'] = False
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                _dict['penalty_yards'] = accepted_pen.yards
                return _dict
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = accepted_pen.yards
            return _dict
        elif len([p for p in pens if p.declined]) == 1 and len([p for p in pens if not p.declined]) == 2: # one declined + two accepted
            pens = [p for p in pens if not p.declined]
            p1, p2 = pens[0], pens[1]
            if any([p1.no_play, p2.no_play]):
                _dict['is_touchdown'] = False
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                _dict['penalty_yards'] = (p1.yards * (-1 if p1.against_possessing_team else 1)) + (p2.yards * (-1 if p2.against_possessing_team else 1))
                return _dict
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = (p1.yards * (-1 if p1.against_possessing_team else 1)) + (p2.yards * (-1 if p2.against_possessing_team else 1))
            print(line)
            print(_dict)
            return _dict
        else:
            print('Missing case for c3:')
            [p.show() for p in pens]
        return _dict
    def penalty_c4(self, pens: list[PenaltyObject], row: pd.Series):
        """
        Get penalty attributes for case 3: EXACTLY 3 PENALTIES
        Args:
            pens (list[PenaltyObject]): penalties
            row (pd.Series): row
        """
        _dict = { col: np.nan for col in self.cols }
        line: str = row['pids_detail']
        if all([p.offset for p in pens]): # all offsetting
            if any([p.no_play for p in pens]): # no play
                _dict['is_touchdown'] = False
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                return _dict
            # offsetting penalties but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        elif all([not p.declined for p in pens]): # all accepted
            if any([p.no_play for p in pens]): # no play
                _dict['is_touchdown'] = False
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                _dict['penalty_yards'] = max([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
                return _dict
            # all accepted but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        # default case (3 accepted, 1 declined, etc.)
        if any([p.no_play for p in pens]): # no play
            _dict['is_touchdown'] = False
            _dict['is_interception'] = False
            _dict['is_spike'] = False
            _dict['penalty_yards'] = max([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        # play stands and yards added
        _dict = self.normal_lambda_func(row, has_penalty=True)
        _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
        return _dict
    def penalty_c6(self, pens: list[PenaltyObject], row: pd.Series):
        """
        Get penalty attributes: EXACTLY 6 PENALTIES
        Args:
            pens (list[PenaltyObject]): penalties
            row (pd.Series): row
        """
        _dict = { col: np.nan for col in self.cols }
        line: str = row['pids_detail']
        if all([p.offset for p in pens]): # all offsetting
            if any([p.no_play for p in pens]): # no play
                _dict['is_touchdown'] = False
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                return _dict
            # offsetting penalties but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        print("New case for penalty_c6")
        return _dict
    def penalty_c8(self, pens: list[PenaltyObject], row: pd.Series):
        """
        Get penalty attributes: EXACTLY 8 PENALTIES
        Args:
            pens (list[PenaltyObject]): penalties
            row (pd.Series): row
        """
        _dict = { col: np.nan for col in self.cols }
        line: str = row['pids_detail']
        if all([p.offset for p in pens]): # all offsetting
            if any([p.no_play for p in pens]): # no play
                _dict['is_touchdown'] = False
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                return _dict
            # offsetting penalties but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        print("New case for penalty_c8")
        return _dict
    def penalty_lambda_func(self, row: pd.Series):
        pens: list[PenaltyObject] = list(set(Penalties().get(row)))
        _dict = {
            1: self.penalty_c1, 2: self.penalty_c2, 3: self.penalty_c3,
            4: self.penalty_c4, 6: self.penalty_c6, 8: self.penalty_c8
        }
        key = len(pens)
        vals = { col: np.nan for col in self.cols }
        try:
            vals = _dict[key](pens, row)
        except KeyError:
            print(f"No dict key for: {key}")
        return vals
    def penalty(self, df: pd.DataFrame):
        """
        Convert lines with penalties
        !!! penalty_yards = NaN and pass_yards = NaN - no play !!!
        Attributes: pass_yards, penalty_yards, penalty_type, is_touchdown, is_interception
        Args:
            df (pd.DataFrame): all
        """
        df[self.cols] = df.apply(lambda x: self.penalty_lambda_func(x), axis=1, result_type='expand')
        return df
    # end penalties
    # challenge
    def challenge_lambda_func(self, row: pd.Series):
        
        return
    def challenge(self, df: pd.DataFrame):
        """
        Convert lines with challenge
        Attributes: pass_yards, penalty_yards, penalty_type, is_touchdown, is_interception
        Args:
            df (pd.DataFrame): all
        """
        df[self.cols] = df.apply(lambda x: self.challenge_lambda_func(x), axis=1, result_type='expand')
        return df
    # end challenge
    def convert(self):
        """
        Convert DF pids_detail to yard/quanities
        """
        atdf = self.all_tables.copy()
        atdf['at_index'] = atdf.index
        df = self.df.merge(atdf[['primary_key', 'down', 'togo', 'at_index']], on=['primary_key'])
        df = df.tail(5000)
        combinations = list(itertools.product([True, False], repeat=len(self.pt_bool_cols)))
        df_list = []
        for comb in combinations:
            condition = (df['is_fumble']==comb[0])&(df['is_penalty']==comb[1])&(df['is_challenge']==comb[2])&(df['is_block']==comb[3])&(df['contains_lateral']==comb[4])
            cd: pd.DataFrame = df.loc[condition]
            func_key = ','.join([self.pt_bool_cols[i] for i in range(len(self.pt_bool_cols)) if comb[i]])
            func_key = 'normal' if func_key == '' else func_key
            if func_key in self.funcs.keys():
                print(f"Calling combination func: {func_key}")
                df_list.append(self.funcs[func_key](cd))
            else:
                # print(f"No func: {func_key}")
                continue
        new_df = pd.concat(df_list)
        new_df = new_df[['pids_detail', 'play_type']+self.pt_bool_cols+self.cols]
        self.save_frame(new_df, "data/temp")
        return
    
##################

Conversion(
    pd.read_csv("%s.csv" % "data/allTables_play_types_data", low_memory=False),
    './'
).convert()

# ---------------------------------------------------------------------------------

# df = pd.read_csv("%s.csv" % "data/allTables_play_types_data", low_memory=False)

# lines = df.loc[df['play_type']=='coin_toss', 'pids_detail'].values

# names = []

# for line in lines:
#     names.append(line.split(" ")[0])
# names = list(set(names))
# names.sort()

# df = pd.DataFrame()
# df['name'] = names
# df.to_csv("%s.csv" % "teamNames_firstName", index=False)
