import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import regex as re

ALL_ENTS = [
    'PASSER', 'RECEIVER', 'RUSHER', 'TACKLER',
    'DEFENDER', 'PENALIZER', 'FUMBLER', 
    'FUMBLE_RECOVERER', 'FUMBLE_FORCER',
    'INTERCEPTOR', 'KICKER', 'PUNTER', 
    'RETURNER', 'SACKER', 'LATERALER',
    'OTHER', 'TEAM_NAME', 'BLOCKER'
]

QB_ENTS = [
    'PASSER', 'RUSHER', 'PENALIZER',
    'FUMBLER', 'LATERALER'
]

def test_passer_epa():
    df = pd.read_csv("%s.csv" % "data/sample")
    cd = pd.read_csv("%s.csv" % "data/sample_entities")
    df = df.merge(cd, on=['primary_key', 'detail'])
    df.fillna('', inplace=True)
    passers = list(set([p.split(":")[0] for p in df.loc[df['PASSER']!='', 'PASSER'].values]))
    # get plays when value in QB_ENTS contains name
    for name in passers:
        mask = df[QB_ENTS].apply(lambda col: col.str.contains(name, case=False, na=False)).any(axis=1)
        temp_df: pd.DataFrame = df[mask]
        print(name, np.mean(temp_df['epa'].values))
    return

def test_poss_epa():
    df = pd.read_csv("%s.csv" % "data/sample")
    cd = pd.read_csv("%s.csv" % "data/sample_possessions")
    edf = pd.read_csv("%s.csv" % "data/sample_entities")
    df = df.merge(cd, on=['primary_key', 'detail'])
    df = df.merge(edf, on=['primary_key'])
    df['dif_epa'] = df.apply(lambda x: x['epa']-x['epb'], axis=1)
    # jl_avg = np.mean(
    #     df.loc[
    #         (~pd.isna(df['dif_epa']))&
    #         (df['detail'].str.contains('Jordan Love')),
    #         'dif_epa'
    #     ].values
    # )
    # print(jl_avg)
    abbrs = list(set(df['possession'].values))
    for abbr in abbrs:
        avg = np.mean(
            df.loc[
                (~pd.isna(df['dif_epa']))&
                (df['possession']==abbr),
                'dif_epa'
            ].values
        )
        print(abbr, avg)
    return

def test_all_dif_epas():
    gd = pd.read_csv("%s.csv" % "../data/gameData")
    gd = gd.loc[gd['wy'].str.contains('2023')]
    df = pd.read_csv("%s.csv" % "data/allTables")
    pdf = pd.read_csv("%s.csv" % "data/allTables_possessions")
    edf = pd.read_csv("%s.csv" % "data/allTables_entities")
    df = df.merge(pdf, on=['primary_key'])
    df = df.merge(edf, on=['primary_key'])
    df = df.loc[df['key'].isin(gd['key'])]
    df['epb'] = pd.to_numeric(df['epb'], errors='coerce')
    df['epa'] = pd.to_numeric(df['epa'], errors='coerce')
    df['dif_epa'] = df.apply(lambda x: x['epa']-x['epb'], axis=1)
    abbrs = list(set(df['possession']))
    new_df = pd.DataFrame(columns=['abbr', 'total_dif_epa'])
    for abbr in abbrs:
        avg = np.mean(
            df.loc[
                (~pd.isna(df['dif_epa']))&
                (df['possession']==abbr)&
                (~pd.isna(df['PASSER'])),
                'dif_epa'
            ].values
        )
        new_df.loc[len(new_df.index)] = [abbr, avg]
    new_df.sort_values(by=['total_dif_epa'], ascending=False, inplace=True)
    print(new_df)
    return

def graph_epas():
    df = pd.read_csv("%s.csv" % "data/features/possession_epas")
    # start = df.loc[df['wy']=='8 | 2023'].index.values[0]
    # df = df.loc[df.index>=start]
    wys = [f'{w} | 2023' for w in range(1, 19)]
    df = df.loc[df['wy'].isin(wys)]
    col_name = 'total_epa'
    home_g = df.groupby('home_abbr').mean()[['home_' + col_name, 'away_' + col_name]]
    home_g = home_g.reset_index()
    home_g.columns = ['abbr', 'off_' + col_name, 'def_' + col_name]
    away_g = df.groupby('away_abbr').mean()[['home_' + col_name, 'away_' + col_name]]
    away_g = away_g.reset_index()
    away_g.columns = ['abbr', 'def_' + col_name, 'off_' + col_name]
    gdf = pd.concat([home_g, away_g])
    gdf = gdf.groupby('abbr').mean()[['off_' + col_name, 'def_' + col_name]]
    gdf.sort_values(by=['def_' + col_name], ascending=True, inplace=True)
    gdf = gdf.reset_index()
    plt.scatter(gdf['off_' + col_name], gdf['def_' + col_name])
    for _, row in gdf.iterrows():
        plt.annotate(row['abbr'], (row['off_' + col_name], row['def_' + col_name]))
    plt.xlabel('off_' + col_name)
    plt.ylabel('def_' + col_name)
    plt.gca().invert_yaxis()
    plt.axhline(y=np.mean(gdf['def_' + col_name]))
    plt.axvline(x=np.mean(gdf['off_' + col_name]))
    plt.show()
    return

def graph_player_epas(position: str, pids: list[str]):
    df = pd.read_csv(f"data/features/{position}_epas.csv")
    for pid in pids:
        temp_df = df.loc[(df['p_id']==pid)&(df['wy'].str.contains('2023'))]
        plt.plot([i for i in range(len(temp_df.index))], temp_df[f'{position}_epa_added'])
    plt.legend(pids)
    plt.show()
    return

def test_play_types_and_yards():
    pid_ent_cols = ALL_ENTS
    pid_ent_cols.remove('TEAM_NAME')
    pid_ent_cols = [f"pid_{col}" for col in pid_ent_cols]
    def get_type(row: pd.Series):
        line = row['pids_detail']
        if pd.isna(line):
            return np.nan
        _type = 'unknown'
        if 'coin toss' in line:
            _type = 'coin toss'
        if not pd.isna(row['pid_PASSER']):
            _type = 'pass'
        if not pd.isna(row['pid_RUSHER']):
            _type = 'run'
        if not pd.isna(row['pid_PUNTER']):
            _type = 'punt'
        if not pd.isna(row['pid_KICKER']):
            if 'kicks off' in line:
                _type = 'kickoff'
            if 'extra point' in line:
                _type = 'exp'
            if 'field goal' in line:
                _type = 'fg'
        if _type == 'unknown' and not pd.isna(row['pid_FUMBLER']):
            _type = 'fumble'
        if 'Penalty on' in line and '(no play)' in line:
            _type = 'penalty'
        if 'challenged' in line:
            _type = 'challenge'
        if 'Timeout' in line:
            _type = 'timeout'
        return _type
    def get_play_type_and_yards(row: pd.Series):
        line = row['pids_detail']
        _type = get_type(row)
        return { 'play_type': _type, 'yards': np.nan }
    # df = pd.read_csv("%s.csv" % "data/sample_pids")
    # pdf = pd.read_csv("%s.csv" % "data/sample_pid_entities")
    # df = df.merge(pdf, on=['primary_key'])
    df = pd.read_csv("%s.csv" % "data/allTables_pids")
    pdf = pd.read_csv("%s.csv" % "data/allTables_pid_entities")
    df = df.merge(pdf, on=['primary_key'])
    df = df.sample(n=1000, random_state=42)
    cols = ['play_type', 'yards']
    df[cols] = df.apply(lambda x: get_play_type_and_yards(x), axis=1, result_type='expand')
    df = df[['primary_key', 'pids_detail', 'play_type']]
    df.to_csv("%s.csv" % "temp", index=False)
    return

#######################

# test_passer_epa()

# test_poss_epa()

# test_all_dif_epas()

# graph_epas()

# position = 'qb'
# pids = ['LoveJo03', 'HurtJa00', 'MahoPa00', 'PurdBr00', 'WilsZa00']
# position = 'rb'
# pids = ['JoneAa00', 'MontDa01', 'SwifDA00', 'HarrNa00', 'AchaDe00']
# position = 'wr'
# pids = ['DoubRo00', 'WickDo00', 'StxxAm00', 'DiggSt00', 'JohnQu02']
# graph_player_epas(position, pids)

test_play_types_and_yards()