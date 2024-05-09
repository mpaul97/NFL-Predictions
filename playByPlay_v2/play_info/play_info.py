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

df = pt.get_df()
keys = [
    '201312010cle-85', '201612250pit-159', '201709240sdg-98', 
    '201711050dal-140', '201712310den-161', '201809160pit-126', 
    '201809160was-36', '201809300dal-147', '201810010den-45', 
    '201810010den-158', '201810070sea-4', '201810110nyg-44', 
    '201810140den-4', '201810140mia-43', '201810140rai-39', 
    '201810210jax-150', '201810210kan-127', '201810210kan-137', 
    '201810210rav-37', '201810210sdg-58', '201810280kan-21', 
    '201811040cle-94', '201811110kan-73', '201811110phi-105', 
    '201811110tam-21', '201811150sea-89', '201812020rai-58', 
    '201812020rai-83', '201812090kan-143', '201812130kan-13', 
    '201812160clt-45', '201812230nor-14', '201812230sea-135', 
    '201812230sea-163', '201901060rav-7', '201909150cin-49', 
    '201909220buf-48', '201909290den-174', '201910060oti-122', 
    '201910060sdg-19', '201910270kan-142', '201911030kan-57', 
    '201911030mia-46', '201911100oti-168', '201911180sdg-16', 
    '201912010den-96', '201912010jax-162', '201912010kan-91', 
    '201912150kan-75', '201912290tam-63', '201912290tam-137', 
    '202001050phi-115', '202009100kan-58', '202009130nwe-17', 
    '202009200sdg-175', '202009200sea-69', '202010040mia-16',
    '202010190buf-29', '202010190buf-52', '202010250cin-107',
    '202011010kan-114', '202011010rav-42', '202011020nyg-9',
    '202011080kan-94', '202011080tam-119', '202011150mia-41',
    '202011150nor-75', '202011160chi-24', '202011220rai-90',
    '202011220rai-133', '202011290buf-15', '202011290gnb-155',
    '202011290min-118', '202011290tam-62', '202011290tam-80',
    '202011290tam-151', '202011300phi-79', '202012060kan-8',
    '202012130mia-50', '202012200crd-195', '202012200rav-63',
    '202101030det-41', '202101030kan-30', '202101160buf-47',
    '202101160buf-122', '202101170kan-103', '202101170nor-85', 
    '202109120atl-115', '202109190car-80', '202109260det-14',
    '202109260det-30', '202109260kan-80', '202110030mia-11',
    '202110100kan-101', '202110170was-157', '202111010kan-113', 
    '202111140rai-91', '202111210chi-169', '202112250crd-139', 
    '202112260kan-73', '202201020chi-111', '202201160kan-31',
    '202201160kan-84', '202209110nyj-30', '202209110nyj-38', 
    '202209250crd-129', '202209250nwe-128', '202210090rav-36',
    '202210090was-149', '202210270tam-49', '202210270tam-50', 
    '202210270tam-67', '202211070nor-41', '202211200rav-41', 
    '202211200rav-116', '202211200rav-125', '202211270nyj-49', 
    '202212040rav-88', '202212110pit-4', '202212110pit-18',
    '202212110pit-119', '202212170cle-6', '202212170cle-74',
    '202212240cle-8', '202212240dal-153', '202212240rav-61', 
    '202212290oti-42', '202301010rav-29', '202301150cin-42',
    '202310010cle-98', '202311190ram-72', '202312030ram-18', 
    '202312030ram-126', '202312100rav-160', '202312170ram-129', 
    '202312210ram-66', '202312300dal-54', '202401130kan-145',
    '202401140det-37', '202401140det-116'
]
df = df.loc[df['primary_key'].isin(keys)]

for index, row in df.iterrows():
    _type = pt.get_play_type(row)
    print(row['pids_detail'])
    print(_type)