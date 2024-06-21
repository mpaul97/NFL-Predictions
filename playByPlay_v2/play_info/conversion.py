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
        self.gd: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../../data/oldGameData_94"))
        self.tn: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../../teamNames/teamNames_firstName"))
        self.tn_pbp: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../../teamNames/teamNames_pbp"))
        self.alt_abbrs: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../../teamNames/altAbbrs"))
        self.pn_df: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "../../playerNames_v2/data/playerInfo"))
        self.all_cols = {
            'pass': ['completed', 'pass_yards', 'is_interception', 'is_spike'],
            'run': ['rush_yards', 'is_qb_run', 'is_wr_run', 'run_direction', 'is_sneak'],
            'sack': ['sack_yards'],
            'kickoff': ['kickoff_yards', 'return_yards', 'is_touchback'],
            'extra_point': ['is_good'],
            'coin_toss': ['kicking_abbr', 'receiving_abbr', 'winning_abbr'],
            'field_goal': ['is_good', 'field_goal_yards'],
            'punt': ['punt_yards', 'return_yards', 'is_fair_catch', 'is_muffed', 'is_touchback'],
            'timeout': ['timeout_abbr', 'timeout_number'],
            'kneel': [],
            'penalty': ['penalty_yards', 'penalty_types', 'penalizers'],
            'challenge': ['challenge_abbr', 'is_successful_challenge']
        }
        self.general_cols = ['is_off_touchdown', 'is_def_touchdown', 'is_safety']
        self.cols = self.general_cols + list(set(self.flatten(list(self.all_cols.values()))))
        self.cols.sort()
        self.normal_play_type_funcs = {
            'pass': self.normal_pass, 'run': self.normal_run, 'sack': self.normal_sack,
            'kickoff': self.normal_kickoff, 'extra_point': self.normal_extra_point, 'coin_toss': self.normal_coin_toss,
            'field_goal': self.normal_field_goal, 'punt': self.normal_punt, 'penalty': self.normal_penalty,
            'kneel': self.normal_kneel, 'timeout': self.normal_timeout
        }
        # all funcs/func_keys
        # self.funcs = {
        #     'normal': self.normal, 'is_penalty': self.penalty, 'is_challenge': self.challenge,
        #     'is_block': self.block, 'contains_lateral': self.lateral
        # }
        self.funcs = {
            'is_fumble': self.fumble
        }
        return
    def flatten(self, l: list):
        return [x for xs in l for x in xs]
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def get_scorer_pid(self, ents: np.ndarray, line: str):
        """
        Gets scorer pid by finding ent closest to 'touchdown'
        EX: Graham Gano 45 yard field goal no good blocked by Juanyeh Thomas, touchdown
        Juanyeh Thomas -> scorer
        Args:
            ents (np.ndarray): array of pid entities
            line (str): pids_detail

        Returns:
            str: pid
        """
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
    def get_run_direction(self, line: str):
        if re.search(r"middle\s", line):
            return 'm'
        if re.search(r"left\send", line):
            return 'le'
        if re.search(r"right\send", line):
            return 're'
        if re.search(r"left\stackle", line):
            return 'lt'
        if re.search(r"right\stackle", line):
            return 'rt'
        if re.search(r"left\sguard", line):
            return 'lg'
        if re.search(r"right\sguard", line):
            return 'rg'
        return np.nan
    def set_off_def_touchdown(self, _dict: dict, row: pd.Series):
        """
        Fills _dict offensive/defensive touchdown fields
        Args:
            row (pd.Series): DF row
            _dict (dict): dict

        Returns:
            dict: _dict
        """
        line: str = row['pids_detail']
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
    def set_safety(self, _dict: dict, row: pd.Series):
        _dict['is_safety'] = ('safety' in row['pids_detail'])
        return _dict
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
        _dict = self.set_safety(_dict, row)
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
        rusher: str = (row['pid_RUSHER']).split(":")[0]
        togo, down = row['togo'], row['down']
        try:
            rusher_pos: str = self.pn_df.loc[self.pn_df['p_id']==rusher, 'positions'].values[0]
            # qb rusher, is center run, yards togo less than or equal to 2, and is 3rd or 4th down
            if 'QB' in rusher_pos:
                _dict['is_qb_run'] = True
                if 'middle' in line and float(togo) <= 2 and float(down) >= 3:
                    _dict['is_sneak'] = True
            # wr rusher
            if 'WR' in rusher_pos:
                _dict['is_wr_run'] = True
        except IndexError:
            pass
        _dict['run_direction'] = self.get_run_direction(line)
        vals: list[str] = re.findall(r"for\s[-]?[0-9]+", line)
        if len(vals) >= 1:
            _dict['rush_yards'] = int(vals[0].replace("for ",""))
        _dict['is_off_touchdown'] = (('touchdown' in line) and ('returned' not in line))
        _dict['is_def_touchdown'] = (('touchdown' in line) and ('returned' in line))
        _dict = self.set_safety(_dict, row)
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
        _dict = self.set_safety(_dict, row)
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
        _dict = self.set_off_def_touchdown(_dict, row)
        _dict = self.set_safety(_dict, row)
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
        _dict = self.set_safety(_dict, row)
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
        _dict = self.set_off_def_touchdown(_dict, row)
        _dict = self.set_safety(_dict, row)
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
        # this is a little different than set_off_def_touchdown
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
        _dict = self.set_safety(_dict, row)
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
        try:
            team_name: str = line.split(" by ")[1]
            abbr: str = self.tn_pbp.loc[self.tn_pbp['names'].str.contains(team_name), 'abbr'].values[0]
            _dict['timeout_abbr'] = abbr
            num: int = int((re.findall(r"\#[0-9]", line)[0]).replace('#',''))
            _dict['timeout_number'] = num
        except IndexError:
            print(f"normal_timeout IndexError: {line}")
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
                return _dict
            # offsetting penalties but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict 
        elif all([not p.declined for p in pens]): # both accepted (always same team)
            _dict['penalty_yards'] = (pens[0].yards + pens[1].yards) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        # one declined, one accepted
        pens = [p for p in pens if not p.declined]
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
                return _dict
            # offsetting penalties but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict 
        elif any([p.no_play for p in pens]): # no play
            return _dict
        elif all([not p.declined for p in pens]): # all accepted (ASSUMING same team) + play stands
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        elif len([p for p in pens if p.declined]) == 2 and len([p for p in pens if not p.declined]) == 1: # two declined + one accepted
            accepted_pen = [p for p in pens if not p.declined][0]
            if any([p.no_play for p in pens]):
                _dict['penalty_yards'] = accepted_pen.yards
                return _dict
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = accepted_pen.yards
            return _dict
        elif len([p for p in pens if p.declined]) == 1 and len([p for p in pens if not p.declined]) == 2: # one declined + two accepted
            pens = [p for p in pens if not p.declined]
            p1, p2 = pens[0], pens[1]
            if any([p1.no_play, p2.no_play]):
                _dict['penalty_yards'] = (p1.yards * (-1 if p1.against_possessing_team else 1)) + (p2.yards * (-1 if p2.against_possessing_team else 1))
                return _dict
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = (p1.yards * (-1 if p1.against_possessing_team else 1)) + (p2.yards * (-1 if p2.against_possessing_team else 1))
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
                _dict['is_interception'] = False
                _dict['is_spike'] = False
                return _dict
            # offsetting penalties but play stands and yards added
            _dict = self.normal_lambda_func(row, has_penalty=True)
            _dict['penalty_yards'] = sum([p.yards for p in pens]) * (-1 if pens[0].against_possessing_team else 1)
            return _dict
        elif all([not p.declined for p in pens]): # all accepted
            if any([p.no_play for p in pens]): # no play
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
        vals['penalty_types'] = '|'.join([p._type for p in pens])
        vals['penalizers'] = '|'.join([p.penalizer for p in pens])
        vals = self.set_safety(vals, row)
        return vals
    def penalty(self, df: pd.DataFrame):
        """
        Convert lines with penalties
        !!! penalty_yards = NaN and pass_yards = NaN - no play !!!
        Attributes: self.cols
        Args:
            df (pd.DataFrame): all
        """
        df[self.cols] = df.apply(lambda x: self.penalty_lambda_func(x), axis=1, result_type='expand')
        return df
    # end penalties
    # challenge
    def challenge_only(self, _dict: dict, row: pd.Series):
        """
        Previous play is TRUE when next line is ONLY challenge.
        Fills challenge_abbr, is_successful_challenge
        EX: 
            McGeRi01 punts 44 yards, touchback + CIN challenged the kick touched ruling, and the original play was overturned.
            => 
            McGeRi01 punts 44 yards, touchback
        Args:
            row (pd.Series): _description_

        Returns:
            _type_: _description_
        """
        line = row['pids_detail']
        abbr = "RA" # replay assistant
        try:
            r_abbr = re.findall(r"[A-Z]{3}", line)[0]
            abbr = self.alt_abbrs.loc[self.alt_abbrs['alt_abbr'].str.contains(r_abbr), 'abbr'].values[0]
        except IndexError:
            pass
        _dict['challenge_abbr'] = abbr
        _dict['is_successful_challenge'] = ('overturned' in line)
        _dict = self.set_safety(_dict, row)
        return _dict
    def challenge_with_play(self, _dict: dict, row: pd.Series):
        line: str = row['pids_detail']
        s1: int = 0
        if re.search(r"Replay\sAssistant", line):
            s1: int = line.index("Replay Assistant")
        else:
            s1: int = re.search(r"[A-Z]{3}\schallenged", line).start()
        org_play: str = line[:s1]
        s2: int = (line.index('overturned.')+len('overturned.')) if 'overturned.' in line else (line.index('upheld.')+len('upheld.'))
        challenge: str = line[s1:s2].lstrip().rstrip()
        row['pids_detail'] = org_play
        if 'overturned.' in line:
            new_play: str = line[s2:].lstrip().rstrip()
            row['pids_detail'] = new_play
        _dict = self.normal_lambda_func(row)
        # info
        abbr = "RA" # replay assistant
        try:
            r_abbr = re.findall(r"[A-Z]{3}", challenge)[0]
            abbr = self.alt_abbrs.loc[self.alt_abbrs['alt_abbr'].str.contains(r_abbr), 'abbr'].values[0]
        except IndexError:
            pass
        _dict['challenge_abbr'] = abbr
        _dict['is_successful_challenge'] = ('overturned' in line)
        return _dict
    def challenge_lambda_func(self, row: pd.Series):
        """
        Challenge lines to normal/challenge attributes conversion
        Args:
            row (pd.Series): DF row
        Returns:
            dict: self.cols
        """
        _dict = { col: np.nan for col in self.cols }
        if row['play_type'] == 'challenge': # only a challenge, no play
            return self.challenge_only(_dict, row)
        # normal play + challenge after
        return self.challenge_with_play(_dict, row)
    def challenge(self, df: pd.DataFrame):
        """
        Convert lines with challenge
        Attributes: self.cols
        Args:
            df (pd.DataFrame): all
        """
        df[self.cols] = df.apply(lambda x: self.challenge_lambda_func(x), axis=1, result_type='expand')
        return df
    # end challenge
    # block
    def block_lambda_func(self, row: pd.Series):
        """
        Block lines to normal/block attributes conversion
        play_type(s): field_goal, extra_point, punt
        Args:
            row (pd.Series): DF row
        Returns:
            dict: self.cols
        """
        # NaN form -> blocked by CrosNi00, touchdown
        _dict = { col: np.nan for col in self.cols } 
        if not pd.isna(row['play_type']): # normal blocked punt, FG, EXP
            _dict = self.normal_lambda_func(row)
        _dict = self.set_off_def_touchdown(_dict, row)
        _dict = self.set_safety(_dict, row)
        return _dict
    def block(self, df: pd.DataFrame):
        """
        Convert lines with block
        Attributes: self.cols
        Args:
            df (pd.DataFrame): all
        """
        df[self.cols] = df.apply(lambda x: self.block_lambda_func(x), axis=1, result_type='expand')
        return df
    # end block
    # lateral
    def lateral_lambda_func(self, row: pd.Series):
        """
        Lateral lines -> IGNORE all lines : default NaN values
        play_type(s): any
        Args:
            row (pd.Series): DF row
        Returns:
            dict: self.cols
        """
        return { col: np.nan for col in self.cols } 
    def lateral(self, df: pd.DataFrame):
        """
        Convert lines with lateral(s)
        Attributes: self.cols
        Args:
            df (pd.DataFrame): all
        """
        df[self.cols] = df.apply(lambda x: self.lateral_lambda_func(x), axis=1, result_type='expand')
        return df
    # end lateral
    # fumble
    def fumble_lambda_func(self, row: pd.Series):
        """
        Fumble lines to attributes : default NaN values
        play_type(s): any
        Args:
            row (pd.Series): DF row
        Returns:
            dict: self.cols
        """
        print(row['pids_detail'])
        return { col: np.nan for col in self.cols } 
    def fumble(self, df: pd.DataFrame):
        """
        Convert lines with fumble(s)
        Attributes: self.cols
        Args:
            df (pd.DataFrame): all
        """
        df[self.cols] = df.apply(lambda x: self.fumble_lambda_func(x), axis=1, result_type='expand')
        return df
    # end fumble
    def convert(self):
        """
        Convert DF pids_detail to yard/quanities
        """
        atdf = self.all_tables.copy()
        atdf['at_index'] = atdf.index
        df = self.df.merge(atdf[['primary_key', 'down', 'togo', 'location', 'at_index']], on=['primary_key'])
        self.df = df.copy()
        # df = df.tail(5000)
        df = df.loc[df['primary_key'].str.contains('201109080gnb')]
        df = df.reset_index(drop=True)
        combinations = list(itertools.product([True, False], repeat=len(self.pt_bool_cols)))
        df_list = []
        for comb in combinations:
            condition = (df['is_fumble']==comb[0])&(df['is_penalty']==comb[1])&(df['is_challenge']==comb[2])&(df['is_block']==comb[3])&(df['contains_lateral']==comb[4])
            cd: pd.DataFrame = df.loc[condition]
            if not cd.empty:
                func_key = ','.join([self.pt_bool_cols[i] for i in range(len(self.pt_bool_cols)) if comb[i]])
                # normal plays
                func_key = 'normal' if func_key == '' else func_key
                # default all lateral plays to contain_lateral func_key, ignore these plays
                func_key = 'contains_lateral' if comb[4] else func_key
                if func_key in self.funcs.keys():
                    print(f"Calling combination func: {func_key}")
                    df_list.append(self.funcs[func_key](cd))
                else:
                    print(f"No func: {func_key}")
                    continue
        new_df = pd.concat(df_list)
        new_df = new_df[['primary_key', 'pids_detail', 'play_type']+self.pt_bool_cols+self.cols]
        self.save_frame(new_df, "data/temp")
        return
    
##################

Conversion(
    pd.read_csv("%s.csv" % "data/allTables_play_types_data", low_memory=False),
    './'
).convert()

# df = pd.read_csv("%s.csv" % "data/allTables_play_types_data", low_memory=False)
# df['pids_detail'].fillna('', inplace=True)
# info = df.loc[df['pids_detail'].str.contains('safety'), 'play_type'].values
    
# for pt in list(set(info)):
#     print(pt)
    # try:
    #     print(df.loc[(df['pids_detail'].str.contains('safety'))&(df['play_type']==pt), 'pids_detail'].values[0])
    # except IndexError:
    #     print(pt, 'empty')
    
# '201909220buf-48'

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
