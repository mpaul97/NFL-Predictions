import pandas as pd
import numpy as np
import os
import random
import regex as re
import itertools

from conversion import Conversion

import sys
sys.path.append("../../")
from pbp_custom_ners.data import PID_ENTS

class PlayInfo:
    def __init__(self, _dir: str):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.main_data_dir = self._dir + "../data/"
        self.labels_dir = self.main_data_dir + "labels/"
        self.pt_cols = ['play_type', 'is_fumble', 'is_penalty', 'is_challenge', 'is_block', 'contains_lateral']
        self.pt_bool_cols = ['is_fumble', 'is_penalty', 'is_challenge', 'is_block', 'contains_lateral']
        self.quantity_funcs = {
            # 'pass': Passes
        }
        return
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def get_df(self):
        df = pd.read_csv("%s.csv" % (self.main_data_dir + "allTables_pids"))
        return df.merge(pd.read_csv("%s.csv" % (self.main_data_dir + "allTables_pid_entities")), on=['primary_key'])
    def get_df_pt(self):
        df = self.get_df()
        return df.merge(pd.read_csv("%s.csv" % (self.data_dir + "allTables_play_types")), on=['primary_key'])
    def get_play_type(self, row: pd.Series):
        _dict = { col: False for col in self.pt_cols }
        _dict['play_type'] = np.nan
        line = row['pids_detail']
        if pd.isna(line):
            return _dict
        # play types
        if 'coin toss' in row['pids_detail']:
            _dict['play_type'] = 'coin_toss'
            return _dict
        if not pd.isna(row['pid_PASSER']):
            _dict['play_type'] = 'pass'
            if 'sacked' in line:
                _dict['play_type'] = 'sack'
            if 'kneels' in line:
                _dict['play_type'] = 'kneel'
        if not pd.isna(row['pid_RUSHER']):
            _dict['play_type'] = 'run'
        if not pd.isna(row['pid_PUNTER']):
            _dict['play_type'] = 'punt'
        if not pd.isna(row['pid_KICKER']):
            if 'kicks off' in line:
                _dict['play_type'] = 'kickoff'
            if 'extra point' in line:
                _dict['play_type'] = 'extra_point'
            if 'field goal' in line:
                _dict['play_type'] = 'field_goal'
        if 'Timeout' in line:
            _dict['play_type'] = 'timeout'
        # other info
        if not pd.isna(row['pid_FUMBLER']):
            _dict['is_fumble'] = True
        if 'Penalty' in line:
            _dict['is_penalty'] = True
        if 'challenged' in line:
            _dict['is_challenge'] = True
        if not pd.isna(row['pid_BLOCKER']):
            _dict['is_block'] = True
        if not pd.isna(row['pid_LATERALER']):
            _dict['contains_lateral'] = True
        # ONLY penalty
        if pd.isna(_dict['play_type']) and _dict['is_penalty']:
            _dict['play_type'] = 'penalty'
        # ONLY challenge
        if pd.isna(_dict['play_type']) and _dict['is_challenge']:
            _dict['play_type'] = 'challenge'
        return _dict
    def build_play_types(self):
        """
        Create play_types using pids_detail and pid_entities
        """
        df = self.get_df()
        # df = df.sample(n=1000, random_state=random.randint(0, 42))
        df[self.pt_cols] = df.apply(lambda x: self.get_play_type(x), axis=1, result_type='expand')
        df = df[['primary_key']+self.pt_cols]
        self.save_frame(df, (self.data_dir + "allTables_play_types"))
        cd = self.get_df_pt()
        self.save_frame(cd, (self.data_dir + "allTables_play_types_data"))
        return
    def build_play_quantities(self):
        """
        Create play_quantities from pids_detail, pid_entities, and play_types
        """
        df = self.get_df_pt()
        pts = list(set(df['play_type'].values))
        pts = [p for p in pts if not pd.isna(p)]
        print(pts)
        # pts = ['pass']
        # for pt in pts:
        #     cd: pd.DataFrame = df.loc[df['play_type']==pt]
        #     self.save_frame(cd, "passes")
        #     # self.quantity_funcs[pt](cd, self.pt_bool_cols).convert()
        return
    
# END / PlayInfo

#######################

pt = PlayInfo("./")

# pt.build_play_types()