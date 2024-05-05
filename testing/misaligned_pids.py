import pandas as pd
import numpy as np
import os

def shift_pids_detail(df: pd.DataFrame, index: int):
    after_details = df.loc[df.index>=index, 'pids_detail'].values
    before_details = df.loc[df.index<index, 'pids_detail'].values
    new_details = list(before_details) + [np.nan] + list(after_details)[:-1]
    df['pids_detail'] = new_details
    return df

PBP_PATH = "../playByPlay_v2/data/"

# keys = ['201112110den', '201112040crd', '201111270sdg', '201111200was']
keys = ['201112040crd']

df = pd.read_csv("%s.csv" % (PBP_PATH + "allTables"))
pids = pd.read_csv("%s.csv" % (PBP_PATH + "allTables_pids"))
new_df = df[['key', 'primary_key', 'detail']].merge(pids, on=['primary_key'])

df_list = []

for key in keys:
    temp_df: pd.DataFrame = df.loc[df['key']==key]
    temp_pids: pd.DataFrame = pids.loc[pids['primary_key'].str.contains(key)]
    # temp_df = temp_df.loc[~temp_df['detail'].str.contains('Overtime')]
    temp_df.reset_index(drop=True, inplace=True)
    tdf = temp_df.merge(temp_pids, on=['primary_key'])
    overtime_start_index = tdf.loc[tdf['quarter']=='Overtime'].index.values[0]
    shift_pids_detail(tdf, overtime_start_index)
    overtime_end_index = tdf.loc[tdf['quarter']=='End of Overtime'].index.values[0]
    shift_pids_detail(tdf, overtime_end_index)
    # ot_kickoff_index = tdf.loc[pd.isna(tdf['quarter'])].index.values[-1]
    # shift_pids_detail(tdf, ot_kickoff_index)
    tdf.to_csv("%s.csv" % "temp1", index=False)