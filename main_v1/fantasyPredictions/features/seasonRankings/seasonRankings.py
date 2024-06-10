import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
import requests
import regex as re
from ordered_set import OrderedSet

INFO = {
    'total_yards': 0,
    'allowed_total_yards': 0,
    'pass_yards': 0,
    'allowed_pass_yards': 0,
    'rush_yards': 0,
    'allowed_rush_yards': 0,
    'pass_touchdowns': 0,
    'allowed_pass_touchdowns': 0,
    'rush_touchdowns': 0,
    'allowed_rush_touchdowns': 0,
    'turnovers': 0,
    'forced_turnovers': 0
}

INFO_SORT_ASCENDING = {
    'total_yards': False,
    'allowed_total_yards': True,
    'pass_yards': False,
    'allowed_pass_yards': True,
    'rush_yards': False,
    'allowed_rush_yards': True,
    'pass_touchdowns': False,
    'allowed_pass_touchdowns': True,
    'rush_touchdowns': False,
    'allowed_rush_touchdowns': True,
    'turnovers': True,
    'forced_turnovers': False
}

def getStats(abbr, wy, cd: pd.DataFrame):
    week = int(wy.split(" | ")[0])
    year = int(wy.split(" | ")[1])
    if week == 1:
        stats: pd.DataFrame = cd.loc[
            (cd['wy'].str.contains(str(year-1)))&
            ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))
        ]
    else:
        try:
            start = cd.loc[cd['wy']==wy].index.values[0]
            stats: pd.DataFrame = cd.loc[
                (cd['wy'].str.contains(str(year)))&
                (cd.index<start)&
                ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))
            ]
        except IndexError:
            stats: pd.DataFrame = cd.loc[
                (cd['wy'].str.contains(str(year)))&
                ((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))
            ]
    home_stats = stats.loc[stats['home_abbr']==abbr]
    away_stats = stats.loc[stats['away_abbr']==abbr]
    all_stats = INFO.copy()
    for name in all_stats:
        if 'allowed' in name:
            all_stats[name] = np.mean(np.concatenate([
                home_stats['away_' + name.replace('allowed_', '')].values,
                away_stats['home_' + name.replace('allowed_', '')].values
            ]))
        elif 'forced' in name:
            all_stats[name] = np.mean(np.concatenate([
                home_stats['away_' + name.replace('forced_', '')].values,
                away_stats['home_' + name.replace('forced_', '')].values
            ]))
        else:
            all_stats[name] = np.mean(np.concatenate([
                home_stats['home_' + name].values,
                away_stats['away_' + name].values
            ]))
    return all_stats

def addRanks(new_df: pd.DataFrame):
    df_list = []
    years = list(OrderedSet([int(wy.split(" | ")[1]) for wy in new_df['wy']]))
    abbrs = { year: list(set(new_df.loc[new_df['wy'].str.contains(str(year)), 'home_abbr'].values)) for year in years }
    wys = list(OrderedSet(new_df['wy']))
    for wy in wys:
        year = int(wy.split(" | ")[1])
        all_abbrs = abbrs[year]
        temp_df: pd.DataFrame = new_df.loc[new_df['wy']==wy]
        start = new_df.loc[new_df['wy']==wy].index.values[0]
        for name in INFO:
            sort_df = pd.DataFrame()
            home_data: pd.DataFrame = temp_df[['home_abbr', ('home_' + name)]].values
            away_data: pd.DataFrame = temp_df[['away_abbr', ('away_' + name)]].values
            curr_abbrs = np.concatenate([home_data[:, 0], away_data[:, 0]])
            curr_vals = np.concatenate([home_data[:, 1], away_data[:, 1]])
            sort_df['abbr'] = curr_abbrs
            sort_df['value'] = curr_vals
            # add abbrs and values of teams on bye - correct ranking
            missing_abbrs = list(set(all_abbrs).difference(set(curr_abbrs)))
            for abbr in missing_abbrs:
                prev_data: pd.DataFrame = new_df.loc[
                    (new_df.index<start)&
                    ((new_df['home_abbr']==abbr)|(new_df['away_abbr']==abbr))
                ].tail(1)
                prefix = 'home_' if prev_data['home_abbr'].values[0] == abbr else 'away_'
                val = prev_data[prefix + name].values[0]
                sort_df.loc[len(sort_df.index)] = [abbr, val]
            sort_dir = INFO_SORT_ASCENDING[name]
            sort_df.sort_values(by=['value'], ascending=sort_dir, inplace=True)
            sort_df.reset_index(drop=True, inplace=True)
            sort_df['rank'] = sort_df.index + 1
            # same rankings for same values
            dup_vals = { val: list(curr_vals).count(val) for val in curr_vals if list(curr_vals).count(val) > 1 }
            for key in dup_vals:
                dup_df = sort_df.loc[sort_df['value']==key]
                min_dup = min(dup_df['rank'])
                sort_df.loc[dup_df.index, 'rank'] = min_dup
            # add to temp_df
            home_vals, away_vals = [], []
            for index, row in temp_df.iterrows():
                home_abbr = row['home_abbr']
                away_abbr = row['away_abbr']
                home_val = sort_df.loc[sort_df['abbr']==home_abbr, 'rank'].values[0]
                away_val = sort_df.loc[sort_df['abbr']==away_abbr, 'rank'].values[0]
                home_vals.append(home_val)
                away_vals.append(away_val)
            temp_df['home_' + name + '_rank'] = home_vals
            temp_df['away_' + name + '_rank'] = away_vals
        df_list.append(temp_df)
    return pd.concat(df_list)

def getNewGameSource(week, year):
    
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/week_' + str(week) + '.htm'
    
    wy = str(week) + " | " + str(year)
    
    print('Building new source: ' + wy + ' ...')

    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    links = []

    # find links
    for link in soup.find_all('a'):
        l = link.get('href')
        if (re.search(r"boxscores\/[0-9]{9}", l) or re.search(r"teams\/[a-z]{3}", l)):
            links.append(l)
            
    if 'teams' in links[0] and 'teams' in links[1]:
        links.pop(0)

    df = pd.DataFrame(columns=['key', 'wy', 'home_abbr', 'away_abbr'])

    # parse links   
    for i in range(0, len(links)-2, 3):
        away_abbr = links[i].split("/")[2].upper()
        home_abbr = links[i+2].split("/")[2].upper()
        key = links[i+1].split("/")[2].replace(".htm","")
        if re.search(r"[0-9]{9}[a-z]{3}", key):
            df.loc[len(df.index)] = [key, wy, home_abbr, away_abbr]
    
    return df

###################################

def buildSeasonRankings(source: pd.DataFrame, _dir):
    if 'seasonRankings.csv' in os.listdir(_dir):
        print('Using existing seasonRankings.csv.')
        return
    print('Creating seasonRankings.csv...')
    rankings_dir = _dir + '../../../gamePredictions/features/seasonRankings/'
    rdf = pd.read_csv("%s.csv" % (rankings_dir + "seasonRankings"))
    # rdf.drop(columns=['home_abbr', 'away_abbr'], inplace=True)
    # source = source.merge(rdf, on=['key', 'wy'])
    # print(source.head())
    new_cols = [col.replace('home_','') for col in rdf.columns if 'home_' in col and 'abbr' not in col]
    new_df = pd.DataFrame(columns=list(source.columns)+new_cols)
    for index, row in source.iterrows():
        print(row['wy'])
        pid = row['p_id']
        key = row['key']
        abbr = row['abbr']
        stats = rdf.loc[rdf['key']==key]
        prefix = 'home_' if (abbr == stats['home_abbr'].values[0]) else 'away_'
        cols = [col for col in rdf.columns if prefix in col and 'abbr' not in col]
        new_df.loc[len(new_df.index)] = list(row.values) + list(stats[cols].values[0])
    new_df.to_csv("%s.csv" % (_dir + "seasonRankings"), index=False)
    return

###################################

def getNewSeasonRankings(source: pd.DataFrame, cd: pd.DataFrame, _dir):
    home_cols = ['home_' + key for key in INFO.keys()]
    away_cols = ['away_' + key for key in INFO.keys()]
    new_df = pd.DataFrame(columns=list(source.columns)+home_cols+away_cols)
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        home_stats = getStats(home_abbr, wy, cd)
        away_stats = getStats(away_abbr, wy, cd)
        new_df.loc[len(new_df.index)] = list(row.values) + list(home_stats.values()) + list(away_stats.values())
    new_df.fillna(new_df.mean(), inplace=True)
    new_df = new_df.round(2)
    new_df = addRanks(new_df)
    new_df.drop_duplicates(inplace=True)
    
    return new_df

def buildNewSeasonRankings(source: pd.DataFrame, cd: pd.DataFrame, _dir):
    print('Creating newSeasonRankings...')
    wy = source['wy'].values[0]
    game_source = getNewGameSource(int(wy.split("|")[0]), int(wy.split("|")[1]))
    # rankings_dir = _dir + '../../../gamePredictions/features/seasonRankings/'
    # rdf = pd.read_csv("%s.csv" % (rankings_dir + "newSeasonRankings_" + wy.replace(' | ', '-')))
    rdf: pd.DataFrame = getNewSeasonRankings(game_source.copy(), cd, _dir)
    new_cols = [col.replace('home_','') for col in rdf.columns if 'home_' in col and 'abbr' not in col]
    new_df = pd.DataFrame(columns=list(source.columns)+new_cols)
    for index, row in source.iterrows():
        pid = row['p_id']
        key = row['key']
        abbr = row['abbr']
        stats = rdf.loc[rdf['key']==key]
        prefix = 'home_' if (abbr == stats['home_abbr'].values[0]) else 'away_'
        cols = [col for col in rdf.columns if prefix in col and 'abbr' not in col]
        new_df.loc[len(new_df.index)] = list(row.values) + list(stats[cols].values[0])
    return new_df

###########################

# source = pd.read_csv("%s.csv" % "../source/source")
# df = pd.read_csv("%s.csv" % "../../../../data/fantasyData")

# buildSeasonRankings(source, df, './')

# source = pd.read_csv("%s.csv" % "../source/new_source")
# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")

# buildNewSeasonRankings(source, cd, './')