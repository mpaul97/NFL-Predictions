import pandas as pd
import numpy as np
import os
import regex as re

import sys
sys.path.append('../')

from pbp_custom_ners.custom_ents import ALL_ENTS

PID_ENTS = ALL_ENTS.copy()
PID_ENTS.remove('TEAM_NAME')

PBP_PATH = "../playByPlay_v2/data/"

def get_pid_entities(row: pd.Series):
    cols = [f"pid_{c}" for c in PID_ENTS]
    detail, pids_detail = row['detail'], row['pids_detail']
    # ents = row[PID_ENTS][~pd.isna(row[PID_ENTS])]
    ents = row[PID_ENTS].to_dict()
    ents = { key: ents[key] for key in ents if not pd.isna(ents[key]) }
    ent_objs = []
    for key in ents:
        val = ents[key]
        for item in val.split("|"):
            name = item.split(":")[0]
            start, end = int(item.split(":")[1]), int(item.split(":")[2])
            # ent_objs.append(PidEnt(name, start, end))
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

def build():
    df = pd.read_csv("%s.csv" % (PBP_PATH + "sample"))
    edf = pd.read_csv("%s.csv" % (PBP_PATH + "sample_entities"))
    pdf = pd.read_csv("%s.csv" % (PBP_PATH + "sample_pids"))
    df = df[['primary_key', 'detail']].merge(edf, on=['primary_key'])
    df = df.merge(pdf, on=['primary_key'])
    cols = [f"pid_{c}" for c in PID_ENTS]
    df[cols] = df.apply(lambda x: get_pid_entities(x), axis=1, result_type="expand")
    df = df[['primary_key']+cols+PID_ENTS]
    df.to_csv("%s.csv" % "temp", index=False)
    return

########################

build()