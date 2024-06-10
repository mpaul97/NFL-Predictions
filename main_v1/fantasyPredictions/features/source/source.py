import pandas as pd
import numpy as np
import os
import regex as re
from ordered_set import OrderedSet
from bs4 import BeautifulSoup
import requests

pd.options.mode.chained_assignment = None

def getSortKey(wy):
    cd = pd.read_csv("%s.csv" % "D:/NFLPredictions3/data/gameData")
    weeks = list(set([int(wy.split(' | ')[0]) for wy in cd['wy'].values]))
    years = list(set([int(wy.split(' | ')[1]) for wy in cd['wy'].values]))
    max_week = max(weeks)
    max_year = max(years)
    t_week = int(wy.split(" | ")[0])
    t_year = int(wy.split(" | ")[1])
    sort_key = t_week/max_week + t_year/max_year
    return sort_key

def buildSource(df: pd.DataFrame, _dir):
    
    # source exists
    if 'source.csv' in os.listdir(_dir):
        print('source already exists in: ' + _dir)
        return pd.read_csv("%s.csv" % (_dir + 'source'))
        
    # source does not exist 
    print('Creating source...')
    
    new_df = pd.DataFrame(columns=['key', 'abbr', 'p_id', 'wy', 'position'])
    
    for index, row in df.iterrows():
        pid = row['p_id']
        position = row['position']
        new_df.loc[len(new_df.index)] = [
            row['game_key'], row['abbr'], pid,
            row['wy'], position
        ]
        
    new_df.sort_values(by=['key', 'abbr'], ascending=True, inplace=True)
    
    new_df.to_csv("%s.csv" % (_dir + "source"), index=False)
    
    return

def getNewGameSource(week, year):
    
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/week_' + str(week) + '.htm'
    
    wy = str(week) + " | " + str(year)
    
    print('Building new game source: ' + wy + ' ...')

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

    df = pd.DataFrame(columns=['key', 'wy', 'abbr'])

    # parse links   
    for i in range(0, len(links)-2, 3):
        away_abbr = links[i].split("/")[2].upper()
        home_abbr = links[i+2].split("/")[2].upper()
        key = links[i+1].split("/")[2].replace(".htm","")
        if re.search(r"[0-9]{9}[a-z]{3}", key):
            df.loc[len(df.index)] = [key, wy, home_abbr]
            df.loc[len(df.index)] = [key, wy, away_abbr]
    
    return df

def buildNewSource(week, year, df: pd.DataFrame, sdf: pd.DataFrame, _dir):
    gdf = getNewGameSource(week, year)
    print('Creating new_source using current week starters...')
    new_wy = str(week) + " | " + str(year)
    df_list = []
    # key,abbr,p_id,wy,position
    off_positions = ['QB', 'RB', 'WR', 'TE']
    for index, row in gdf.iterrows():
        key = row['key']
        abbr = row['abbr']
        wy = row['wy']
        if wy in df['wy'].values:
            info: pd.DataFrame = df.loc[(df['abbr']==abbr)&(df['wy']==wy)]
            info.drop(columns=['points', 'week_rank'], inplace=True)
            info['key'] = key
            info['wy'] = new_wy
            info = info[['key', 'abbr', 'p_id','wy', 'position']]
            df_list.append(info)
        else:
            starters = (sdf.loc[sdf['abbr']==abbr, 'starters'].values[0]).split("|")
            temp_df = pd.DataFrame(columns=['key', 'abbr', 'p_id', 'wy', 'position'])
            for s in starters:
                pid = s.split(":")[0]
                position = s.split(":")[1]
                if position in off_positions:
                    temp_df.loc[len(temp_df.index)] = [key, abbr, pid, new_wy, position]
            df_list.append(temp_df)
    new_df = pd.concat(df_list)
    new_df.to_csv("%s.csv" % (_dir + "new_source"), index=False)
    return new_df

########################

# week = 1
# year = 2022
# df = pd.read_csv("%s.csv" % "../../../../data/fantasyData")
# sdf = pd.read_csv("%s.csv" % "../../../../data/starters_23/starters_w1")

# buildNewSource(week, year, df, sdf, './')

# --------------------------------------------
# drop 2002 superbowl
# key = '200301260rai'

# source = pd.read_csv("%s.csv" % "source")
# drops = source.loc[source['key']==key].index.values

# source.drop(drops, inplace=True)

# source.to_csv("%s.csv" % "source", index=False)
